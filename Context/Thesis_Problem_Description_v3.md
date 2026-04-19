# **Dynamic Batch Roasting Scheduling with Shared Pipeline Constraints under Unplanned Disruptions: A Case Study at Nestlé Trị An**

\---

## **1. Tổng Quan Bài Toán**

Bài toán đặt ra là xây dựng và đánh giá một hệ thống **reactive scheduling** (lập lịch phản ứng) cho quá trình rang cà phê theo batch tại nhà máy Nestlé Trị An, trong phạm vi một ca làm việc kéo dài 8 giờ, được discretize thành **480 time slot, mỗi slot = 1 phút** (slot 0 đến slot 479). Đây là bài toán lập lịch đa máy, đa sản phẩm trên 5 unrelated parallel roasters thuộc 2 production lines.

Hệ thống phải đồng thời giải quyết hai bài toán con lồng vào nhau:

* **Bài toán 1 — MTO (Make-to-Order):** Đáp ứng các đơn hàng NDG và Busta với due date mềm tại nửa ca (slot 240). Số lượng batch MTO nhỏ (3–5 batch) nhưng bắt buộc hoàn thành, và chỉ chạy được trên một số roaster nhất định (R1/R2 cho NDG, R2 cho Busta), tạo ra **resource contention** trực tiếp với production PSC trên cùng roaster.
* **Bài toán 2 — MTS (Make-to-Stock):** Duy trì tồn kho RC (Roasted Coffee) cho PSC ở mức ổn định bằng cách liên tục lập lịch các batch PSC trên tất cả 5 roasters, đáp ứng consumption rate cố định của dây chuyền đóng gói PSC. Nếu RC stock về 0, dây chuyền đóng gói dừng (**stockout**). Nếu RC stock đầy, roaster không có chỗ đổ output và phải dừng (**overflow**). Cả hai đều là failure mode phải phòng tránh.

Hai bài toán con này lồng vào nhau vì MTO batch chiếm roaster và pipeline — mỗi phút R1 hoặc R2 dành cho NDG/Busta là một phút không rang PSC, trực tiếp làm giảm RC supply rate. Scheduler phải cân bằng: đáp ứng MTO đúng hạn vs. duy trì RC buffer cho PSC.

**Ràng buộc vật lý cốt lõi** là **shared GC pipeline per line** — mỗi line có một đường ống dùng chung mà tất cả roaster trên line đó phải cạnh tranh quyền truy cập. Đường ống phục vụ thao tác **consume** (lấy green coffee cho batch), kéo dài 3 phút mỗi lần, và **không cho phép overlap**. Trên Line 2 với 3 roasters, pipeline utilization đạt **60%**, chỉ còn 6 phút slack per cycle — đủ chặt để bất kỳ disruption nào cũng gây cascading delay sang các roaster khác.

**Yếu tố stochastic chính** là **Unplanned Stoppages (UPS)** — sự cố thiết bị ngoài kế hoạch xảy ra theo phân phối xác suất, không biết trước tại thời điểm lập lịch ban đầu. Khi UPS xảy ra giữa ca, batch đang rang trên roaster bị ảnh hưởng **bị hủy hoàn toàn** (GC đã consume mất, phải restart từ đầu nếu muốn), và hệ thống phải re-schedule phần còn lại của ca dựa trên trạng thái hiện tại. Đây là thách thức trung tâm của luận văn: **làm thế nào để reactive scheduling strategy tốt nhất phục hồi throughput và duy trì RC stock sau disruption?**

Luận văn so sánh bốn chiến lược reactive scheduling — dispatching heuristic (baseline, đại diện thao tác operator hiện tại), tabular Q-Learning (RL đơn giản nhất), MaskedPPO (end-to-end Deep RL), và RL-based Hyper-Heuristic (DRL có cấu trúc, agent chọn từ tập heuristic tools đơn giản) — trên một thiết kế thí nghiệm factorial có kiểm soát, thay đổi cường độ disruption để xác định điều kiện mà mỗi chiến lược chiếm ưu thế. Bốn phương pháp tạo thành một **method ladder** từ không học → học đơn giản → học sâu → học có cấu trúc, kiểm tra liệu thêm complexity có thêm value hay không. MILP và CP-SAT được sử dụng như **deterministic benchmark** — xác lập performance ceiling lý thuyết (~$295k profit). CP-SAT không tham gia reactive experiments vì thời gian re-solve (~2 phút trên phần cứng thesis) không phù hợp cho real-time.

Mục tiêu tối ưu là **maximize profit** — tổng revenue từ batch hoàn thành trừ đi các chi phí phạt: MTO tardiness, setup cost (changeover), RC stockout, roaster idle khi stock thấp, và overflow-idle. Mọi thành phần đều quy về cùng đơn vị tiền ($), cho phép so sánh trực tiếp giá trị giữa các chiến lược và làm reward function cho DRL agent. Chi tiết cấu trúc chi phí xem tại `cost.md`.

\---

## **2. Cấu Hình Hệ Thống Vật Lý**

### **2.1 Hai Line Sản Xuất**

Nhà máy có hai production lines hoạt động song song.

**Line 1** bao gồm **Roaster 1** và **Roaster 2**, với RC buffer riêng (aggregate batch counter).

**Line 2** bao gồm **Roaster 3**, **Roaster 4** và **Roaster 5**, với RC buffer riêng.

Hai line **độc lập hoàn toàn** về RC inventory và pipeline — pipeline Line 1 chỉ phục vụ R1, R2; pipeline Line 2 phục vụ R3, R4, R5. Ngoại trừ **Roaster 3** — điểm kết nối chéo duy nhất: R3 luôn consume GC từ Line 2 pipeline, nhưng RC output có thể đổ vào RC stock của Line 1 hoặc Line 2 tùy quyết định scheduler (xem §2.2).

Hệ quả: nếu không có R3 cross-line routing, hai line có thể được giải quyết **hoàn toàn độc lập** — hai bài toán scheduling riêng biệt nhỏ hơn. Với R3 routing, hai line **coupled** thông qua R3, tạo ra bài toán scheduling chung không thể tách rời.

### **2.2 Năm Roasters — Eligibility, Pipeline Mapping, và RC Output**

Mỗi roaster xử lý tối đa **1 batch** tại một thời điểm. Thời gian rang **phụ thuộc SKU**: $p_{PSC} = 15$ phút, $p_{NDG} = 17$ phút, $p_{BUS} = 18$ phút. Sự khác biệt thời gian rang giữa các SKU tạo ra hiệu ứng **desynchronization** — các roaster ban đầu stagger đều 3 phút sẽ dần lệch nhịp sau vài batch mixed SKU, làm pipeline scheduling trở nên phức tạp hơn đáng kể.

#### **Chi tiết từng roaster:**

**Roaster 1 (R1) — Line 1, MTO+MTS capable:**

* SKU eligible: **PSC, NDG**
* Pipeline consume: **Line 1 pipeline** (3 phút per batch)
* RC output: **luôn → Line 1 RC stock** (fixed, không phải decision variable)
* Vai trò vận hành: R1 là 1 trong 2 roaster duy nhất có thể rang NDG. Khi có NDG order, R1 phải chia sẻ capacity giữa NDG (MTO, bắt buộc) và PSC (MTS, duy trì stock). Nếu NDG chiếm nhiều slot, Line 1 RC supply giảm → risk stockout trên Line 1.

**Roaster 2 (R2) — Line 1, most flexible:**

* SKU eligible: **PSC, NDG, Busta**
* Pipeline consume: **Line 1 pipeline** (3 phút per batch)
* RC output: **luôn → Line 1 RC stock** (fixed)
* Vai trò vận hành: R2 là roaster duy nhất rang được Busta. Nếu có cả NDG và Busta order, R2 bắt buộc phải rang Busta (không ai khác làm được), và NDG phải dồn hết cho R1. Setup time giữa Busta → PSC và NDG → PSC đều tốn 5 phút, nên lịch trình R2 đặc biệt nhạy cảm với thứ tự batch.

**Roaster 3 (R3) — Line 2, cross-line bridge:**

* SKU eligible: **chỉ PSC**
* Pipeline consume: **luôn Line 2 pipeline** (3 phút per batch), bất kể RC output đi đâu
* RC output: **Line 1 hoặc Line 2** — đây là **decision variable** ($y\_b \\in {0,1}$), scheduler quyết định cho từng batch
* Vai trò vận hành: R3 là roaster đặc biệt nhất trong hệ thống. Về mặt pipeline, R3 **luôn cạnh tranh với R4 và R5** trên Line 2 pipeline — bất kể output đi đâu. Nhưng về mặt inventory, R3 có thể "cứu" Line 1 khi stock thấp bằng cách route output về Line 1. Tradeoff: mỗi batch R3 route về Line 1 = 1 batch RC cho Line 1, nhưng vẫn chiếm 3 phút pipeline Line 2, giảm throughput capacity cho R4/R5.
* **Trong thiết kế thí nghiệm:** R3 routing là **experimental factor** — so sánh "fixed" (R3 luôn output → Line 2, hai line decoupled) vs. "flexible" (R3 output là decision variable, hai line coupled). Mục đích: đo lường giá trị thực sự của cross-line flexibility dưới các mức disruption khác nhau.

**Roaster 4 (R4) — Line 2, dedicated PSC:**

* SKU eligible: **chỉ PSC**
* Pipeline consume: **Line 2 pipeline** (3 phút per batch)
* RC output: **luôn → Line 2 RC stock** (fixed)
* Vai trò vận hành: Pure PSC roaster. Không bao giờ phải setup (chỉ rang PSC), nên throughput lý thuyết đạt tối đa. Chia sẻ pipeline với R3 và R5.

**Roaster 5 (R5) — Line 2, dedicated PSC:**

* Giống hệt R4.

#### **Bảng tóm tắt:**

|Roaster|Line|SKU eligible|Pipeline consume|RC output|Đặc điểm|
|-|-|-|-|-|-|
|R1|1|PSC, NDG|Line 1|Line 1 (fixed)|MTO capable, shared capacity|
|R2|1|PSC, NDG, Busta|Line 1|Line 1 (fixed)|Most flexible, Busta-only capable|
|R3|2|PSC|Line 2 (always)|**Line 1 or 2 (decision)**|Cross-line bridge|
|R4|2|PSC|Line 2|Line 2 (fixed)|Dedicated PSC|
|R5|2|PSC|Line 2|Line 2 (fixed)|Dedicated PSC|

> **Thời gian rang phụ thuộc SKU:** PSC = 15 phút, NDG = 17 phút, Busta = 18 phút. Pipeline consume luôn = 3 phút bất kể SKU.

### **2.3 Sequence-Dependent Setup Time — Ràng Buộc Cứng**

Khi hai batch **liên tiếp trên cùng một roaster** có **SKU khác nhau**, bắt buộc phải có **5 phút (5 time slot) setup time** giữa thời điểm kết thúc batch trước và thời điểm bắt đầu batch sau.

**Quy tắc chi tiết:**

* Setup áp dụng cho **mọi loại transition SKU** mà không phân biệt mức độ khác nhau: PSC → NDG, NDG → PSC, PSC → Busta, Busta → PSC, NDG → Busta, Busta → NDG — **tất cả đều tốn đúng 5 phút** (Assumption A2).
* Nếu hai batch liên tiếp **cùng SKU**, không có setup time — batch sau có thể bắt đầu ngay (chỉ cần pipeline rảnh).
* Trong 5 phút setup, roaster **không làm gì** — không rang, không consume GC. Roaster ở trạng thái SETUP.
* Đường ống GC **không bị chiếm** bởi setup time — pipeline tự do phục vụ consume cho roaster khác trên cùng line.
* Setup time là **hard constraint** — không thể rút ngắn, không thể bỏ qua.
* Mỗi lần setup phát sinh **chi phí cố định** $c^{setup} = \$800$, tính tại thời điểm bắt đầu setup — không phải per-minute. Chi phí này cộng thêm vào chi phí cơ hội (lost throughput) và safety-idle penalty. Xem `cost.md` §3.4.

**Ví dụ minh họa chi tiết trên R2:**

```
R2 schedule: Busta batch kết thúc tại t=80. Batch tiếp theo là PSC.

Vì Busta ≠ PSC → setup required.

t=80:  Busta batch ends. R2 enters SETUP state.
t=81:  SETUP (1/5)
t=82:  SETUP (2/5)
t=83:  SETUP (3/5)
t=84:  SETUP (4/5) — trong lúc này, R1 có thể consume trên pipeline Line 1 bình thường
t=85:  SETUP complete. R2 now IDLE, ready to start PSC batch.
t=85:  R2 bắt đầu PSC batch (nếu pipeline rảnh). Consume: t=85→t=87. Roast: t=85→t=99.
```

Nếu batch tiếp theo sau Busta vẫn là Busta (cùng SKU):

```
t=80:  Busta batch ends. R2 immediately IDLE (no setup).
t=80:  R2 có thể bắt đầu Busta batch tiếp theo ngay (nếu pipeline rảnh).
```

**Hệ quả chiến lược:** Scheduler bị khuyến khích tự nhiên **nhóm tất cả batch cùng SKU liên tiếp** trên cùng roaster. Mỗi lần xen kẽ SKU tốn 5 phút idle + $800 chi phí setup — chi phí kép này trực tiếp giảm throughput VÀ lợi nhuận. Ví dụ: nếu R1 cần rang 3 batch NDG và 10 batch PSC, lịch tối ưu là \[3× NDG] → \[setup 5 min, $800] → \[10× PSC], chỉ tốn 1 lần setup = 5 phút + $800. Nếu xen kẽ NDG-PSC-NDG-PSC-NDG-PSC..., tốn 5 lần setup = 25 phút + $4,000 mất trắng.

**Trạng thái SKU ban đầu (đầu ca):** Tất cả 5 roaster bắt đầu ca với $\\text{last\_sku} = k^{PSC}$ — nghĩa là roaster ở trạng thái "đã rang PSC lần cuối". Hệ quả:

* Batch PSC đầu tiên trên bất kỳ roaster nào: **không cần setup** (cùng SKU)
* Batch NDG đầu tiên trên R1 hoặc R2: **cần 5 phút setup** \[0, 5) trước khi bắt đầu
* Batch Busta đầu tiên trên R2: **cần 5 phút setup** \[0, 5) trước khi bắt đầu
* Nếu R1 rang NDG đầu ca: NDG batch sớm nhất bắt đầu tại slot 5. Nếu R3 rang PSC: bắt đầu ngay slot 0.

### **2.4 Shared GC Pipeline — Ràng Buộc Vật Lý Cốt Lõi**

#### **2.4.1 Cấu trúc pipeline**

Mỗi line có **một đường ống GC (Green Coffee) duy nhất** kết nối nguồn GC với tất cả roaster trên line đó:

* **Pipeline Line 1:** phục vụ R1 và R2
* **Pipeline Line 2:** phục vụ R3, R4 và R5

Hai pipeline hoạt động **hoàn toàn độc lập** — Line 1 pipeline và Line 2 pipeline không ảnh hưởng lẫn nhau.

#### **2.4.2 Thao tác trên pipeline: Consume và Restock**

Pipeline thực hiện hai loại thao tác:

**Consume** — lấy green coffee beans từ GC silo vào roaster khi batch bắt đầu. Mỗi thao tác consume **kéo dài 3 phút** (3 time slot), bất kể loại SKU hay lượng GC. GC được lấy từ silo tương ứng trên line đó (xem §2.6 GC Silo).

**Restock** — nạp green coffee từ big bag vào GC silo. Mỗi thao tác restock **kéo dài 15 phút** và **chiếm toàn bộ pipeline** trong suốt thời gian đó. Trong 15 phút restock, không roaster nào trên line đó có thể bắt đầu batch mới (vì consume cần pipeline). Restock thêm **5 batch** vào silo được chọn. Chỉ **1 restock tại 1 thời điểm** trên toàn nhà máy (shared restock station — xem §2.6).

#### **2.4.3 Mutual Exclusion — NoOverlap Constraint**

**Ràng buộc cốt lõi:** Tại bất kỳ thời điểm nào, pipeline của mỗi line chỉ phục vụ **tối đa 1 consume operation**. Không có exception.

Nếu pipeline đang bận phục vụ roaster A (đang trong 3 phút consume), roaster B phải **đợi** cho đến khi pipeline rảnh — ngay cả khi roaster B đã sẵn sàng (idle, không cần setup). Thời gian đợi pipeline gây ra **idle time ngoài kiểm soát** của roaster — đây là nguồn delay chính trong hệ thống.

#### **2.4.4 Consume Timing — Song Song Với Đầu Quá Trình Rang**

Thời điểm batch bắt đầu rang **đồng thời** là thời điểm consume bắt đầu. Pipeline bận trong **3 phút đầu** của batch, trong khi roaster bận trong **toàn bộ thời gian rang** ($p_k$ phút, tùy SKU). Hai khoảng thời gian này **overlap** — consume là hoạt động song song với đầu quá trình rang, **không phải** giai đoạn tuần tự trước khi rang.

```
Timeline chi tiết cho 1 batch PSC bắt đầu tại t=0 (p_PSC = 15):

Pipeline:  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
            t=0  t=1  t=2  (3 slots bận)

Roaster:   ████████████████████████████████████████████████████████████
            t=0  t=1  t=2  t=3  ...  t=13  t=14  (15 slots bận)

           |---consume (3 min)---|
           |--------------------roasting (15 min)--------------------|
           t=0                  t=3                                 t=15

Timeline cho 1 batch NDG bắt đầu tại t=0 (p_NDG = 17):

           |---consume (3 min)---|
           |----------------------roasting (17 min)----------------------|
           t=0                  t=3                                     t=17

Pipeline trở lại FREE tại t=3 → roaster khác trên cùng line có thể bắt đầu consume tại t=3.
PSC roaster trở lại IDLE tại t=15. NDG roaster trở lại IDLE tại t=17.
Busta roaster trở lại IDLE tại t=18.
```

Điều này có nghĩa: ngay khi pipeline rảnh (sau 3 phút), roaster khác có thể bắt đầu batch tiếp theo **trong khi** roaster hiện tại vẫn đang rang. Đây là lý do nhiều batch có thể **overlap trên các roaster khác nhau** miễn là pipeline không conflict.

#### **2.4.5 Pipeline Utilization Analysis — Tại Sao Line 2 Là Bottleneck**

**Line 1 (R1, R2) — 2 roasters:**

Trong điều kiện lý tưởng (không setup, không downtime), mỗi roaster cần pipeline 3 phút mỗi batch cycle (PSC: 15 min, NDG: 17 min, Busta: 18 min). Pipeline utilization phụ thuộc SKU mix:

* Pipeline demand per PSC cycle: 2 roasters × 3 min = **6 min per 15-min cycle**
* **Utilization (all PSC): 6/15 = 40%**
* Slack: **9 min** — rộng rãi. Hai roaster hiếm khi conflict.
* Khi rang NDG (17 min) hoặc Busta (18 min): batch cycle dài hơn → pipeline utilization giảm nhẹ → nhưng stagger pattern bị phá vỡ khi mixed SKU.

Ví dụ lịch trình không conflict:

```
Pipeline Line 1:
R1: ███...............███...............███...............
R2: ...███...............███...............███............
    t=0  3  6  9  12 15 18 21 24 27 30 33 36 39 42 45

R1 consume: \\\[0,2], \\\[15,17], \\\[30,32], ...
R2 consume: \\\[3,5], \\\[18,20], \\\[33,35], ...
→ Không overlap. Cả 2 roaster chạy ở max throughput.
```

**Line 2 (R3, R4, R5) — 3 roasters:**

* Pipeline demand: 3 roasters × 3 min = **9 min per 15-min cycle**
* Pipeline capacity: 15 min
* **Utilization: 9/15 = 60%**
* Slack: **6 min** — chỉ 6 phút dư mỗi chu kỳ.

Ví dụ lịch trình lý tưởng (tight nhưng khả thi):

```
Pipeline Line 2:
R3: ███............███............███............
R4: ...███............███............███.........
R5: ......███............███............███......
    t=0  3  6  9  12 15 18 21 24 27 30 33 36 39

R3 consume: \\\[0,2], \\\[15,17], \\\[30,32], ...
R4 consume: \\\[3,5], \\\[18,20], \\\[33,35], ...
R5 consume: \\\[6,8], \\\[21,23], \\\[36,38], ...
→ Không overlap. Nhưng nếu BẤT KỲ roaster nào bị delay 1 slot,
   consume sẽ chen vào slot của roaster tiếp theo → cascade.
```

**Cascading delay scenario trên Line 2:**

Giả sử R3 bị delay 2 phút (do setup hoặc UPS recovery):

```
Original:     R3:\\\[0,2]   R4:\\\[3,5]   R5:\\\[6,8]
After delay:  R3:\\\[2,4]   → conflict với R4 tại slot 3,4
              R4 phải shift: R4:\\\[5,7]   → conflict với R5 tại slot 6,7  
              R5 phải shift: R5:\\\[8,10]
→ R3 delay 2 min → R4 delay 2 min → R5 delay 2 min = DOMINO EFFECT
→ Tổng throughput loss: 2 phút per roaster = 6 phút total pipeline delay
```

Đây là lý do Line 2 pipeline là **ràng buộc cốt lõi** của bài toán — nhỏ nhất disruption cũng cascade.

#### **2.4.6 Interaction Giữa Setup Time và Pipeline Contention**

Setup time và pipeline contention là **hai nguồn delay độc lập** có thể cộng dồn:

* **Setup time:** Roaster chờ 5 phút sau khi đổi SKU. **Không chiếm pipeline.** Roaster khác trên cùng line có thể consume bình thường trong lúc roaster đang setup.
* **Pipeline contention:** Roaster chờ pipeline rảnh để bắt đầu consume. **Không liên quan đến setup.**

**Cộng dồn:** Nếu R1 cần setup (5 phút) VÀ khi hết setup, pipeline đang bận phục vụ R2 (3 phút), R1 phải chờ setup + pipeline = tối đa 5 + 3 = 8 phút idle trước khi bắt đầu batch tiếp theo.

Ví dụ cụ thể:

```
R1: Batch NDG ends at t=40. Next batch: PSC (khác SKU → setup 5 min).
    t=40–44: SETUP (pipeline free, R2 có thể dùng)
    t=45: Setup done. R1 ready to consume.
    Nhưng: R2 đang consume từ t=44 đến t=46 (bắt đầu batch PSC lúc t=44).
    → R1 phải đợi pipeline đến t=47.
    t=47: R1 bắt đầu PSC batch. Consume: \\\[47,49]. Roast: \\\[47,61].
    
    Total idle cho R1: t=40 → t=47 = 7 phút (5 setup + 2 pipeline wait).
```

### **2.5 RC Inventory — Aggregate Batch Counter Per Line**

#### **2.5.1 Cách theo dõi inventory**

RC (Roasted Coffee) inventory được theo dõi dưới dạng **một số nguyên** cho mỗi line, tính bằng **số batch** (không phải kg). Mỗi line có một bộ đếm RC stock duy nhất — không phân biệt từng RC silo riêng lẻ.

Ký hiệu: $B\_l(t)$ = RC stock trên line $l$ tại time slot $t$ (đơn vị: batch).

#### **2.5.2 RC stock thay đổi bởi hai sự kiện**

**RC tăng (+1) khi:** Một batch **PSC** hoàn thành rang trên roaster có RC output thuộc line $l$. Batch hoàn thành tại thời điểm $t = s\_b + 15$ (start time + processing time). Tại thời điểm đó, $B\_l$ tăng 1 đơn vị.

Lưu ý: chỉ **PSC batch** mới tăng RC stock. NDG và Busta batch **không đổ vào RC** — sản phẩm MTO được giao thẳng cho khách, không qua inventory. Do đó, mỗi phút roaster dành cho MTO là một phút **không bổ sung RC stock** cho line đó.

**RC giảm (−1) khi:** PSC consumption event xảy ra theo lịch cố định (xem §3 PSC Consumption). Mỗi event tại thời điểm $\\tau \\in \\mathcal{E}\_l$ giảm $B\_l$ đi 1 đơn vị.

#### **2.5.3 Inventory balance equation**

$$B\_l(t) = B^0\_l + \\underbrace{\\sum\_{\\substack{b \\in \\mathcal{B}^{PSC}:\\ \\text{out}(b) = l,; e\_b \\leq t}} 1}\_{\\text{PSC batches completed on line } l \\text{ by time } t} ;;-;; \\underbrace{\\left|\\left{ \\tau \\in \\mathcal{E}*l : \\tau \\leq t \\right}\\right|}*{\\text{consumption events on line } l \\text{ by time } t}$$

Trong đó $B^0\_l$ là RC stock ban đầu (initial condition, input).

#### **2.5.4 Inventory bounds — hard constraints**

**Stockout — $B\_l(t) < 0$:** Nghĩa là consumption event xảy ra nhưng không có RC batch nào để lấy. Dây chuyền PSC packaging **dừng** — đây là failure mode chính cần phòng tránh.

* Trong **deterministic mode** (lập lịch ban đầu, không có UPS): $B\_l(t) \\geq 0$ là **hard constraint**. Model phải tìm schedule sao cho RC không bao giờ về 0 khi có consumption event.
* Trong **reactive mode** (sau UPS, re-solve): stockout có thể **unavoidable** (ví dụ: 2 roasters trên Line 1 cùng down 30 phút, consumption vẫn tiếp tục). Lúc này $B\_l(\\tau) \\geq 0$ tại consumption events chuyển thành **soft constraint** với penalty $c^{stock} = $1{,}500$ **per consumption event** bị thiếu hàng — model tìm schedule **minimizes số stockout events** thay vì tuyên bố infeasible. Xem `cost.md` §3.2.

**Overflow — $B\_l(t) > \\overline{B}\_l$:** Nghĩa là roaster hoàn thành PSC batch nhưng RC stock đã đầy — không có chỗ chứa output. Roaster **phải dừng** — không thể xả RC. Đây là **hard constraint trong mọi mode** vì overflow là physical impossibility (silo đầy vật lý, không thể ép thêm).

$\\overline{B}\_l$ = maximum RC buffer capacity per line = $20{,}000 \\text{ kg} / 500 \\text{ kg/batch} = $ **40 batches per line**.

**Safety stock threshold** $\\theta^{SS} = 20$ **batches** (nửa max\_buffer). Khi $B\_l(t) < 20$, roaster idle trên line đó bị phạt $c^{idle} = $200$/phút — khuyến khích scheduler giữ roaster hoạt động khi stock thấp. Xem `cost.md` §3.4 cho chi tiết.

**Overstock-idle:** Khi $B\_l(t) = 40$ (đầy) và roaster không thể start batch PSC vì không có chỗ đổ output, roaster bị **forced idle** với penalty $c^{over} = $50$/phút — penalty nhẹ vì đây chỉ là inefficiency, không phải crisis. Xem `cost.md` §3.3.

#### **2.5.5 Không có SKU mixing issue**

Mỗi ca chỉ sản xuất **1 PSC SKU duy nhất**. Không có PSC-PSC SKU switch. RC inventory luôn chứa single-SKU. Do đó, không cần theo dõi SKU assignment trong RC, không cần switching logic, không cần threshold conditions.

#### **2.5.6 Ví dụ RC inventory trajectory**

Giả sử Line 1, $B^0\_{L1} = 10$ batch, consumption rate $\\rho = 5$ phút/batch (1 batch consumed mỗi 5 phút), 2 roasters (R1, R2) cả hai đang rang PSC liên tục:

```
Thời gian  | Events                          | B\\\_L1
-----------+---------------------------------+------
t=0        | Start of shift                  | 10
t=5        | Consumption event               | 9
t=10       | Consumption event               | 8
t=15       | R1 batch#1 done (+1), consume   | 8    (= 8+1-1)
t=17       | R2 batch#1 done (+1)            | 9
t=20       | Consumption event               | 8
t=25       | Consumption event               | 7
t=30       | R1 batch#2 done (+1), consume   | 7    (= 7+1-1)
t=32       | R2 batch#2 done (+1)            | 8
...

Steady state: mỗi 15 phút, R1 và R2 mỗi roaster produce 1 batch = +2.
             Consumption trong 15 phút: 15/5 = 3 events = −3.
             Net rate: +2 − 3 = −1 per 15 minutes.
→ RC stock giảm dần 1 batch mỗi 15 phút.
→ Nếu ban đầu 10, sau 150 phút stock ≈ 0 → STOCKOUT RISK.
```

**Quan trọng:** Ví dụ trên cho thấy với 2 roasters (Line 1) và rate 5 phút/batch, **RC stock giảm dần** — roasters không sản xuất đủ nhanh để bù consumption. Đây là trường hợp cần R3 route output về Line 1 để hỗ trợ. Với R3 support: +3 batches per 15 min vs. −3 consumption = balanced. Nhưng R3 support = chiếm Line 2 pipeline, giảm throughput Line 2.

\---

### **2.6 GC Silo — Quản Lý Nguyên Liệu Đầu Vào**

#### **2.6.1 Cấu trúc GC Silo**

Green Coffee (GC) được lưu trữ trong các **silo chuyên dụng** theo SKU và theo line. Mỗi silo có **dung lượng hữu hạn** (tính bằng batch) và cần được **restock** định kỳ khi cạn.

**Bảng cấu hình GC Silo:**

|Line|SKU|Dung lượng (batch)|Mức ban đầu (batch)|Ghi chú|
|-|-|-|-|-|
|L1|PSC|40|20|Phục vụ R1, R2|
|L1|NDG|10|5|Phục vụ R1, R2 (khi rang NDG)|
|L1|Busta|10|5|Phục vụ R2 (khi rang Busta)|
|L2|PSC|40|20|Phục vụ R3, R4, R5|

> **Tại sao Line 2 chỉ có PSC silo:** Cả ba roaster trên Line 2 (R3, R4, R5) chỉ eligible rang PSC. Không cần NDG hay Busta GC trên Line 2.

**GC silo mapping — roaster nào lấy từ silo nào:**

Roaster $r$ rang SKU $k$ → consume GC từ silo $(\text{pipe}(r), k)$.
```
R1 rang NDG  → consume từ GC silo (L1, NDG)
R2 rang Busta → consume từ GC silo (L1, Busta)  
R3 rang PSC  → consume từ GC silo (L2, PSC) — bất kể RC output đi đâu
R4 rang PSC  → consume từ GC silo (L2, PSC)
```

#### **2.6.2 GC Silo Balance**

$$G_{l,k}(t) = G^0_{l,k} + 5 \times \text{restocks\_completed}_{l,k}(t) - \text{batches\_started}_{l,k}(t)$$

GC silo giảm 1 mỗi khi batch SKU $k$ **bắt đầu** trên roaster thuộc line $l$ (GC consumed ngay khi batch start). GC silo tăng 5 mỗi khi restock hoàn thành.

**Hard constraint:** $G_{l,k}(t) \geq 0$ — không thể rang nếu silo trống. Đây là ràng buộc vật lý tuyệt đối — không có "soft" mode.

#### **2.6.3 Restock — Nạp Lại GC**

**Thông số restock:**
* Mỗi lần restock nạp **5 batch** GC vào silo được chọn
* Thời gian restock: **15 phút**, chiếm toàn bộ pipeline của line đó
* Chỉ **1 restock tại 1 thời điểm** trên toàn nhà máy (shared restock station — nhân công manual loading từ big bag)
* **Không có chi phí trực tiếp** — chi phí duy nhất là opportunity cost (pipeline bị khóa 15 phút)
* Restock chỉ được phép nếu silo **còn chỗ**: $G_{l,k}(t) + 5 \leq \overline{G}_{l,k}$

**Restock blocking effect:**

Trong 15 phút restock trên line $l$, pipeline $l$ bị chiếm hoàn toàn. Roasters đang giữa batch (đã qua giai đoạn consume 3 phút) **tiếp tục rang bình thường**. Nhưng khi batch hoàn thành, roaster **không thể bắt đầu batch mới** vì consume cần pipeline — phải đợi restock kết thúc.

```
Restock PSC trên Line 2 bắt đầu tại t=100, kết thúc tại t=115.

R3: batch hoàn thành tại t=105. Muốn bắt đầu batch mới.
  → Pipeline bận (restock). KHÔNG THỂ BẮT ĐẦU.
  → R3 idle từ t=105 đến t=115 (10 phút forced idle).

R4: batch hoàn thành tại t=112.
  → Pipeline vẫn bận. R4 idle 3 phút.

R5: batch hoàn thành tại t=114.  
  → R5 idle 1 phút.

Sau t=115: pipeline tự do. Ba roaster cạnh tranh pipeline bình thường.
GC silo (L2, PSC) tăng thêm 5 batch tại t=115.
```

**Shared restock station — cross-line coupling:**

Chỉ 1 restock tại 1 thời điểm toàn nhà máy. Nếu Line 1 đang restock, Line 2 phải đợi.

```
t=100: Line 1 cần restock PSC (silo ở 3 batch).
       Line 2 cần restock PSC (silo ở 2 batch — urgent hơn).

Nếu Line 2 restock trước: [100, 115) → Line 1 phải đợi → restock tại [115, 130).
  Rủi ro: Line 1 PSC silo có thể cạn trước t=115 nếu consumption nhanh.

Nếu Line 1 restock trước: [100, 115) → Line 2 phải đợi → restock tại [115, 130).
  Rủi ro: Line 2 PSC silo cạn trước t=115 → 3 roasters đồng loạt idle.

→ Scheduler phải quyết định ai restock trước dựa trên urgency và tác động downstream.
   Greedy chọn line có silo thấp hơn; CP-SAT/DRL có thể tìm timing tối ưu hơn.
```

#### **2.6.4 GC Silo và Scheduling Tension**

GC silo tạo ra **ba nguồn scheduling tension mới**:

1. **Timing tension:** Khi nào restock? Sớm = lãng phí pipeline time khi chưa cần. Muộn = rủi ro silo cạn, tất cả roasters trên line forced idle cho đến khi restock xong.

2. **Cross-line competition:** Shared restock station = hai line cạnh tranh quyền restock. Scheduling phải coordinate across lines.

3. **Interaction với UPS:** Nếu UPS xảy ra trong khi restock đang chạy → roaster bị UPS không rang được (đã down), VÀ các roaster khác trên cùng line cũng không rang được (pipeline bận restock). Double disruption.

\---

## **3. PSC Consumption**

### **3.1 Consumption Rate**

PSC tiêu thụ RC theo lịch **deterministic** với tốc độ cố định, đặc trưng cho từng PSC SKU và từng line.

**Consumption rate** $\\rho\_l$: Mỗi line tiêu thụ 1 batch RC mỗi $\\rho\_l$ phút. Giá trị $\\rho\_l$ là **input** của bài toán, được xác định bởi PSC SKU đang chạy trên line đó.

Ví dụ: $\\rho\_l = 5.1$ phút nghĩa là dây chuyền PSC packaging của line $l$ lấy 1 batch RC từ stock khoảng mỗi 5.1 phút.

### **3.2 Consumption Schedule — Precomputed Input**

Từ rate $\\rho\_l$, tính trước **danh sách thời điểm consumption events** trước khi bắt đầu solve:

$$\\mathcal{E}\_l = \\left{ \\lfloor i \\cdot \\rho\_l \\rfloor ;\\middle|; i = 1, 2, \\dots, \\left\\lfloor \\frac{480}{\\rho\_l} \\right\\rfloor \\right}$$

**Ví dụ chi tiết:** $\\rho\_l = 5.1$ phút:

```
i=1:  floor(1 × 5.1) = floor(5.1)  = 5
i=2:  floor(2 × 5.1) = floor(10.2) = 10
i=3:  floor(3 × 5.1) = floor(15.3) = 15
i=4:  floor(4 × 5.1) = floor(20.4) = 20
i=5:  floor(5 × 5.1) = floor(25.5) = 25
i=6:  floor(6 × 5.1) = floor(30.6) = 30
i=7:  floor(7 × 5.1) = floor(35.7) = 35
i=8:  floor(8 × 5.1) = floor(40.8) = 40
i=9:  floor(9 × 5.1) = floor(45.9) = 45
i=10: floor(10 × 5.1) = floor(51.0) = 51  ← gap nhảy từ 45 lên 51 (6 min thay vì 5)
...
i=94: floor(94 × 5.1) = floor(479.4) = 479  ← event cuối cùng trong ca

Tổng: 94 events per shift cho rate 5.1 phút.
```

Mỗi event tại thời điểm $\\tau \\in \\mathcal{E}\_l$ giảm $B\_l$ đi 1 đơn vị.

### **3.3 Tính chất quan trọng**

* Consumption rate **không đổi** trong suốt ca.
* **Chỉ 1 PSC SKU** per line per shift — không có PSC-PSC SKU switch, không có thay đổi rate giữa ca.
* Consumption xảy ra **bất kể** tình trạng scheduling — dây chuyền đóng gói chạy liên tục và không quan tâm roasters đang làm gì. Nếu RC stock = 0 khi consumption event đến, đây là **stockout**.
* Consumption schedule là **input cố định** — scheduler không thể thay đổi hay trì hoãn consumption. Scheduler chỉ có thể điều chỉnh **production schedule** (khi nào rang batch nào) để đảm bảo RC luôn đủ.

\---

## **4. Sản Phẩm, Jobs và Batches**

### **4.1 NDG và Busta — Make-to-Order (MTO)**

NDG và Busta là sản phẩm sản xuất **theo đơn hàng**. Toàn bộ thông tin MTO đã biết từ đầu ca:

* **Loại SKU:** NDG hoặc Busta (hai SKU riêng biệt)
* **Số batch cần rang:** Thường tổng cộng **3–5 batch** cho cả NDG và Busta trong ca
* **Roaster eligibility:**

  * NDG: rang được trên **R1 hoặc R2** (scheduler chọn)
  * Busta: **chỉ R2** (không có roaster nào khác rang được)

**MTO Jobs:** Mỗi MTO order (ví dụ: "4 batch NDG") là 1 **job** gồm nhiều batch. Tất cả batch của cùng job có cùng SKU. Job hoàn thành khi batch cuối cùng của job hoàn thành.

**Due date — soft constraint:** Tất cả batch MTO phải hoàn thành trước **slot 240** (nửa ca = 4 giờ). Đây là **soft constraint** với penalty $c^{tard} = $1{,}000$/phút trễ (xem `cost.md` §3.1):

$$\\text{tard}*j = \\max\\left(0,;; \\max*{b \\in \\mathcal{B}\_j} (s\_b + p_{k_b}) - 240\\right)$$

Tardiness tính theo batch **cuối cùng** của job — nếu batch NDG cuối hoàn thành tại slot 250 ($s_b + 17 = 250$), tardiness = 10 phút.

Lý do dùng soft constraint (không hard): trong tình huống cực đoan — ví dụ UPS trên R2 kéo dài 45 phút trong giờ đầu ca khi có Busta order (chỉ R2 rang được) — việc đáp ứng due date có thể physically impossible. Hard constraint sẽ khiến model báo infeasible và không đưa ra schedule nào. Soft constraint cho phép model tìm **schedule tốt nhất có thể** và báo cáo mức tardiness.

**RC output:** NDG và Busta **không đổ vào RC stock**. Sản phẩm MTO hoàn thành được giao thẳng, không qua inventory. MTO batch chỉ ảnh hưởng đến scheduling:

* Chiếm roaster ($p_k$ phút: NDG = 17, Busta = 18)
* Chiếm pipeline (3 phút)
* Có thể gây setup time (nếu roaster phải chuyển từ PSC → NDG hoặc ngược lại)

Nhưng **không ảnh hưởng đến RC balance** — RC stock chỉ tăng khi PSC batch hoàn thành.

### **4.2 PSC — Make-to-Stock (MTS)**

PSC là sản phẩm rang để **bổ sung RC inventory**. Không có ticket cố định — scheduler tự quyết định:

* **Bao nhiêu batch PSC** cần rang trong ca (decision variable — không biết trước)
* **Trên roaster nào** (tất cả 5 roasters đều eligible cho PSC)
* **Tại thời điểm nào** (phải cân bằng: rang đủ sớm để tránh stockout, không rang quá nhiều gây overflow)

Mỗi ca chỉ sản xuất **1 PSC SKU duy nhất** — không có PSC-PSC SKU switch.

PSC batch count là **decision variable** — model sử dụng **pool of optional batches** (pre-generated) và solver chọn bao nhiêu batch cần activate. Pool size = theoretical maximum = $\\lfloor 480/15 \\rfloor = 32$ per roaster × 5 roasters = 160 optional PSC batches. Trong thực tế, solver chỉ activate \~50–80 batch (phụ thuộc vào consumption rate và MTO demand).

### **4.3 Tương tác giữa MTO và MTS — Resource Contention**

Đây là tension trung tâm của bài toán:

* R1 và R2 phải **chia capacity** giữa MTO (bắt buộc, có deadline) và PSC (tự nguyện, nhưng cần để tránh stockout).
* Mỗi batch NDG trên R1/R2 = 17 phút roaster + 3 phút pipeline **không rang PSC**. Mỗi batch Busta = 18 phút + 3 phút pipeline. Thời gian MTO dài hơn PSC (15 phút) → tốn capacity nhiều hơn.
* Nếu NDG/Busta order lớn (5 batch), R1/R2 dành \~85–90 phút + setup time cho MTO → Line 1 thiếu PSC supply → cần R3 route output về Line 1 để bù.
* Setup time khi chuyển NDG → PSC (5 phút) thêm chi phí cho mỗi lần xen kẽ.

Ví dụ cụ thể:

```
Ca có: 4 batch NDG (R1 hoặc R2) + 1 batch Busta (R2 only).
Consumption rate Line 1: 1 batch / 5 phút = 96 batches needed per shift.

R2 phải rang: 1 Busta + PSC còn lại
R1 phải rang: 4 NDG + PSC còn lại

Thời gian MTO trên Line 1: 4 × 17 (NDG) + 1 × 18 (Busta) + 2 × 5 (setup) = 96 phút
                                                    (NDG→PSC + Busta→PSC)
Thời gian còn lại cho PSC trên Line 1: (2 × 480) − 96 = 864 phút roaster-time
PSC batches from Line 1: 864 / 15 ≈ 57 batches
Consumption events: 96 batches needed
Gap: 96 − 57 = 39 batches phải đến từ R3 routing

So với uniform 15 phút: MTO chiếm 96 phút thay vì 85 phút → 11 phút thêm
→ ~1 PSC batch ít hơn → R3 routing pressure tăng.
```

\---

## **5. Planned Downtime**

### **5.1 Định nghĩa**

Planned downtime là các khoảng thời gian đã lên lịch trước mà roaster không thể hoạt động. Thông tin này là **input cố định**, biết từ đầu ca, dưới dạng tập hợp time slots cho mỗi roaster:

$$\\mathcal{D}\_r \\subset \\mathcal{T} = {0, 1, \\dots, 479} \\quad \\forall r \\in \\mathcal{R}$$

Downtime có thể khai báo:

* **Per roaster:** Chỉ R3 down trong \[100, 130]
* **Per line:** R1 và R2 cùng down trong \[200, 250]
* **Toàn nhà máy:** Tất cả R1-R5 down trong \[300, 320]

### **5.2 Quy tắc chi tiết**

**Quy tắc 1 — Không được start batch nếu không thể hoàn thành trước downtime:**

Batch chỉ được start tại slot $t$ trên roaster $r$ nếu toàn bộ khoảng processing $\[t, t+14]$ (15 slots) **không overlap** với $\\mathcal{D}\_r$.

$$\[s\_b, s\_b + 14] \\cap \\mathcal{D}\_r = \\emptyset \\quad \\forall b : r\_b = r$$

Nếu downtime bắt đầu tại slot 100, batch cuối cùng có thể start là tại slot 85 (kết thúc tại slot 99). Roaster phải idle từ slot 86 (nếu slot 85 đã bị chiếm) cho đến khi downtime kết thúc.

**Quy tắc 2 — Không có mid-batch pause/resume cho planned downtime:**

Scheduler biết trước downtime windows và **phải** lập lịch tránh chúng. Không có tình huống batch "bị cắt ngang" bởi planned downtime — đó là lỗi scheduling.

**Quy tắc 3 — Sau downtime kết thúc:**

Roaster trở lại trạng thái **IDLE** và có thể start batch mới bình thường. Không cần setup — **trừ khi** batch mới có SKU khác với batch cuối cùng trước downtime (thì phải setup 5 phút bình thường).

**Ví dụ:**

```
R4 planned downtime: \\\[120, 180] (1 giờ, từ phút 120 đến phút 180)
R4 đang rang PSC.

Batch cuối trước downtime:
  → Phải start trước slot 106 (106 + 14 = 120, vừa kịp hoàn thành trước downtime)
  → Nếu batch trước kết thúc tại slot 105, có thể start batch mới tại 105
  → Nhưng 105 + 14 = 119 < 120 → OK
  → Nếu batch trước kết thúc tại slot 108: 108 + 14 = 122 > 120 → KHÔNG ĐƯỢC START
  → R4 phải idle từ 108 đến 180. Total idle: 72 phút (12 phút trước downtime + 60 phút downtime).

Sau downtime:
  → t=180: R4 IDLE. Batch tiếp theo là PSC (cùng SKU) → không cần setup.
  → t=180: R4 start PSC batch (nếu pipeline rảnh). Consume: \\\[180,182]. Roast: \\\[180,194].
```

\---

## **6. Unplanned Stoppages (UPS) — Yếu Tố Stochastic Trung Tâm**

UPS là sự cố thiết bị ngoài kế hoạch, **không biết trước** tại thời điểm lập lịch ban đầu. UPS là **yếu tố stochastic trung tâm** của luận văn — toàn bộ reactive scheduling framework được xây dựng xung quanh việc xử lý UPS.

### **6.1 UPS Event Model**

Mỗi UPS event được đặc trưng bởi ba tham số:

* **Thời điểm xảy ra $t^{UPS}$:** Phân phối Exponential inter-arrival với rate $\\lambda$ (expected number of UPS events per shift).

  * $\\lambda = 0$: Không có UPS (deterministic baseline)
  * $\\lambda = 1$: Trung bình 1 UPS per shift (nhẹ)
  * $\\lambda = 3$: Trung bình 3 UPS per shift (trung bình)
  * $\\lambda = 5$: Trung bình 5 UPS per shift (nặng)
* **Duration $d$:** Thời gian roaster DOWN, drawn from phân phối tham số hóa với mean $\\mu$ phút.

  * $\\mu = 10$ phút: Sửa nhanh (thay bộ phận nhỏ)
  * $\\mu = 20$ phút: Sửa trung bình
  * $\\mu = 30$ phút: Sửa lớn (thay bộ phận chính)
* **Affected roaster:** Random uniform trên 5 roasters. Mỗi UPS ảnh hưởng **đúng 1 roaster** (Assumption A6 — không model correlated failures).

**Lưu ý quan trọng:** UPS parameters là **synthetic** — dựa trên ước lượng MTBF/MTTR từ literature, **không calibrate** từ dữ liệu Nestlé Trị An cụ thể. Đây là limitation được acknowledge. Thiết kế thí nghiệm factorial trên grid ($\\lambda$, $\\mu$) bù đắp bằng cách bao phủ **dải rộng** disruption intensity — kết quả cho thấy **relative performance** giữa strategies, không phải absolute prediction cho nhà máy.

### **6.2 UPS Impact — Batch Bị Hủy Hoàn Toàn**

Khi UPS xảy ra tại thời điểm $t^{UPS}$ trên roaster $r$:

**Trường hợp 1: Roaster $r$ đang rang batch $b$ (RUNNING state):**

1. **Batch $b$ bị hủy hoàn toàn.** Quá trình rang dừng ngay lập tức. Batch không thể salvage, không có partial output. Cà phê rang dở bên trong roaster bị coi là **waste**.
2. **GC đã consume cho batch $b$ mất.** 3 phút pipeline đã dùng cho consume ở đầu batch là **sunk cost** — pipeline time đã trôi, không lấy lại được. GC đã lấy từ silo cũng mất — silo level đã giảm khi batch bắt đầu và **không được hoàn trả** khi batch bị hủy.
3. **RC output không được tính.** Batch chưa hoàn thành → RC stock **không tăng** → consumption vẫn tiếp tục → RC giảm → tăng stockout risk.
4. **Roaster $r$ chuyển sang DOWN state** trong $d$ phút. Trong suốt thời gian DOWN, roaster không thể làm gì.

**Trường hợp 2: Roaster $r$ đang IDLE hoặc SETUP:**

1. Không có batch nào bị hủy (không có batch đang chạy).
2. Roaster $r$ chuyển sang **DOWN state** trong $d$ phút.
3. Nếu đang SETUP: setup timer **reset** — khi roaster trở lại, nếu batch tiếp theo vẫn cần setup, phải setup lại từ đầu 5 phút.

**Trường hợp 3: Roaster $r$ đang DOWN (từ UPS trước):**

Nếu UPS xảy ra lần nữa trên roaster đã down → duration **cộng dồn** hoặc **replace** (tùy convention; đề xuất: lấy max(remaining\_down, new\_duration) — tức là nếu UPS mới dài hơn, extend thời gian down).

### **6.3 Sau Khi UPS Kết Thúc — Scheduler Decision**

Khi roaster $r$ trở lại service (DOWN → IDLE), đây là một **decision point** — scheduler phải quyết định:

* **Có restart batch $b$ (batch bị hủy) hay không?** Đây là **decision variable**, không phải automatic restart.

  * Nếu **restart**: cần consume lại (3 phút pipeline) + rang lại toàn bộ ($p_k$ phút). Đây là batch **mới hoàn toàn** — pipeline phải available, tốn thêm 3 phút pipeline time.
  * Nếu **skip**: roaster tự do chọn batch khác (có thể khác SKU → thêm 5 phút setup).
* **Considerations cho quyết định restart:**

  * Nếu batch $b$ là **MTO** (NDG/Busta): skip = tăng tardiness penalty. Nếu due date gần, restart gần như bắt buộc.
  * Nếu batch $b$ là **PSC**: skip = mất 1 batch throughput, nhưng có thể schedule batch PSC khác ở slot tốt hơn. Restart không nhất thiết tối ưu nếu pipeline đang busy.
  * **RC stock level tại thời điểm đó:** Nếu stock rất thấp, ưu tiên rang PSC ngay (bất kể restart hay batch mới). Nếu stock comfortable, có thể rang MTO trước.

### **6.4 UPS Cascading Effect — Ví Dụ Chi Tiết Trên Line 2**

Đây là ví dụ minh họa tại sao UPS đặc biệt disruptive trên Line 2 (60% pipeline utilization):

```
Trạng thái bình thường trước UPS:
  R3: batch hoàn thành tại t=100, consume tiếp theo tại t=100: \\\[100,102]
  R4: batch hoàn thành tại t=101, consume tiếp theo tại t=103: \\\[103,105]
  R5: batch hoàn thành tại t=103, consume tiếp theo tại t=106: \\\[106,108]
  → Pipeline schedule tốt: R3→R4→R5 interleaved, 1 phút gap mỗi lần.

UPS trên R4 tại t=95, duration = 20 phút:
  1. t=95: R4 batch đang rang bị HỦY. R4 DOWN đến t=115.
  2. t=100: R3 hoàn thành batch, consume bình thường \\\[100,102]. ✓
  3. t=103: R5 hoàn thành batch.
     → Lẽ ra R4 consume tại \\\[103,105] nhưng R4 down.
     → R5 consume tại \\\[103,105]. ✓ (R4 slot bỏ trống, R5 lấy)
  4. t=115: R4 trở lại. Scheduler quyết định restart batch PSC.
     → R4 cần consume. Pipeline status tại t=115?
     → R3 consume cuối: \\\[115,117] (R3 có batch mới start tại t=115)
     → CONFLICT! R4 phải đợi đến t=118.
     → R4 consume: \\\[118,120]. Batch start: t=118, done: t=132.
  5. Cascade:
     → R4 consume tại \\\[118,120] → chen vào slot dự kiến của R5
     → R5 phải shift consume từ \\\[118,...] sang \\\[121,...]
     → Total delay: R4 mất 20 phút (UPS) + 3 phút (pipeline wait)
                     R5 mất 3 phút (pipeline cascade)
     → Throughput loss: 1 batch hủy (R4) + \\\~1-2 batches delay (cascade)
```

### **6.5 UPS vs. Planned Downtime — So Sánh Chi Tiết**

|Thuộc tính|Planned Downtime|Unplanned Stoppage (UPS)|
|-|-|-|
|**Biết trước?**|Có — input cố định từ đầu ca|Không — random event theo phân phối xác suất|
|**Scheduler có thể phòng tránh?**|Có — lập lịch tránh trước (hard constraint)|Không — chỉ có thể react sau khi xảy ra|
|**Mid-batch behavior**|Batch không start nếu không kịp hoàn thành|Batch đang rang bị **hủy hoàn toàn**, GC mất|
|**Pipeline impact khi xảy ra**|Không (scheduler đã tránh)|Consume trước đó là sunk cost; re-consume cần 3 phút pipeline mới|
|**Sau khi kết thúc**|Roaster IDLE, start batch bình thường|Scheduler **quyết định** restart batch bị hủy hay chuyển sang batch khác|
|**RC inventory impact**|Predictable (scheduler biết trước idle time)|Unpredictable (mất 1 batch output + thêm idle time)|
|**Trong model**|Hard constraint: $\[s\_b, e\_b) \\cap \\mathcal{D}\_r = \\emptyset$|Event trigger: re-solve CP-SAT / RL action / dispatching rule|

\---

## **7. Objective Function — Maximize Profit**

> Chi tiết đầy đủ về cấu trúc chi phí, calibration, và ví dụ: xem `cost.md`.

### **7.1 Thiết kế: Mọi thứ quy về tiền ($)**

Objective function là **Maximize Profit** — tổng revenue trừ tổng costs, tất cả cùng đơn vị USD ($). Đây là thiết kế tốt hơn "maximize batches minus weighted penalties" vì:

* Mọi thành phần cùng đơn vị → không cần calibrate trọng số tương đối
* DRL reward function map trực tiếp → agent tối ưu hóa chính xác cái ta quan tâm
* Dễ diễn giải → "schedule này tạo $280,000 profit per shift" có ý nghĩa thực tế

### **7.2 Revenue — Batch Hoàn Thành**

|SKU|Revenue per batch|Ghi chú|
|-|-|-|
|PSC|**$4,000**|Make-to-stock. Revenue earned khi batch hoàn thành ($t = e\_b$)|
|NDG|**$7,000**|Make-to-order. 1.75× giá trị PSC — sản phẩm specialty cao cấp hơn|
|Busta|**$7,000**|Make-to-order. Cùng giá trị NDG|

Chỉ **completed** batches earn revenue. Batch bị UPS hủy giữa chừng: revenue = $0.

### **7.3 Costs — Các Loại Phạt**

|Loại phạt|Ký hiệu|Giá trị|Trigger|Priority|
|-|-|-|-|-|
|**MTO Tardiness**|$c^{tard}$|**$1,000/phút**|MTO job hoàn thành sau slot 240|2nd|
|**RC Stockout**|$c^{stock}$|**$1,500/event/line**|Consumption event xảy ra khi $B\_l(\tau) < 0$ (strictly negative — demand unmet). $B\_l = 0$ is NOT stockout. Phạt **per event**, không phải per minute.|**1st (cao nhất)**|
|**Safety-Idle**|$c^{idle}$|**$200/phút/roaster**|Roaster idle (không RUNNING, không DOWN) khi $B\_{\\ell(r)}(t) < \\theta^{SS} = 20$|3rd|
|**Overflow-Idle**|$c^{over}$|**$50/phút/roaster**|Roaster forced idle vì RC đầy. R1/R2/R4/R5: $B\_{\\text{out}(r)}(t) = 40$. **R3: chỉ khi CẢ HAI line đều = 40** (vì R3 có thể route sang line còn trống).|4th (thấp nhất)|

**Hierarchy:** Stockout ($1,500/event) > Tardiness ($1,000/min) > Safety-Idle ($200/min) > Overflow-Idle ($50/min)

### **7.4 Deterministic Mode (Initial Schedule)**

$$\\boxed{\\text{Maximize Profit} = \\sum\_{b: a\_b=1} R\_{\\text{sku}(b)} - c^{tard} \\sum\_j \\text{tard}*j - c^{idle} \\sum*{r,t} \\text{idle}*{r,t} - c^{over} \\sum*{r,t} \\text{over}\_{r,t}}$$

Không có $c^{stock}$ trong deterministic mode — stockout là hard constraint ($B\_l(t) \\geq 0$), solver không bao giờ gặp stockout.

### **7.5 Reactive Mode (Post-UPS Re-solve)**

$$\\boxed{\\text{Max Profit}*{rem} = \\sum*{b \\in \\mathcal{B}*{rem}} R*{\\text{sku}(b)} \\cdot a\_b - c^{tard}\\sum\_j \\text{tard}*j - c^{stock}\\sum\_l \\text{SO}l - c^{idle}\\sum{r,t} \\text{idle}*{r,t} - c^{over}\\sum\_{r,t} \\text{over}\_{r,t}}$$

Trong đó:

* $\text{SO}\_l = |\{\tau \in \mathcal{E}\_l^{rem} : B\_l(\tau) < 0\}|$ — **số lượng consumption events** trên line $l$ mà stock strictly negative (demand unmet). $B\_l = 0$ không phải stockout. Đây là **event-count**, không phải duration.
* $c^{stock} = $1{,}500$ per event — phạt mỗi **lần** consumption event bị thiếu hàng, không phải mỗi **phút** thiếu hàng.
* Tất cả 4 thành phần cost đều active vì UPS có thể gây stockout và overflow-idle unavoidable.

**Phân biệt penalty vs. KPI (quan trọng):**

* **Penalty (trong objective/reward):** $c^{stock} \times \text{SO}\_l$ — số lượng consumption events với $B\_l < 0$ (strictly negative). Đây là metric mà solver/agent **tối ưu hóa**.
* **KPI (báo cáo, không tối ưu hóa):** "Stockout duration" = tổng phút $B\_l \leq 0$ (bao gồm cả zero — line stall chờ stock hồi phục). Đây là metric **vận hành** dùng để phân tích kết quả. Hai metric tương quan nhưng **không tương đương**.

### **7.6 Ví dụ so sánh schedule**

|Schedule|PSC done|MTO tard|Idle(RC<20)|Over-idle|Revenue|Costs|**Profit**|
|-|-|-|-|-|-|-|-|
|A|65|0 min|30 min|0 min|$288k|$6k|**$282k**|
|B|68|8 min|10 min|5 min|$300k|$10.25k|**$289.75k**|
|C|60|0 min|80 min|0 min|$268k|$16k|**$252k**|

Schedule B wins — extra PSC batches ($12k) outweigh tardiness cost ($8k). Schedule C is worst — too much idle time under low stock.

\---

## **8. Decision Variables**

Model tự quyết định toàn bộ các biến sau tại mỗi lần solve (initial hoặc re-solve):

* **Batch activation (PSC only):** $a\_b \\in {0,1}$ — batch PSC nào cần activate từ pool of optional batches. MTO batches luôn active ($a\_b = 1$).
* **Roaster assignment:** $r\_b \\in \\mathcal{R}\_b$ — mỗi batch chạy trên roaster nào, giới hạn bởi eligibility (NDG → R1/R2, Busta → R2, PSC → R1-R5).
* **Start time:** $s\_b \\in \[0, 465]$ — mỗi batch bắt đầu tại time slot nào. Giới hạn trên 465 đảm bảo batch hoàn thành trước hết ca ($465 + 15 = 480$). Phải tôn trọng:

  * Setup time 5 phút nếu đổi SKU (hoặc nếu là batch đầu ca khác PSC)
  * Pipeline availability (3 phút consume không overlap)
  * Planned downtime clearance (batch phải hoàn thành trước downtime)
  * RC overflow prevention ($B\_l(e\_b) \\leq 40$)
* **R3 RC output direction:** $y\_b \\in {0,1}$ — mỗi batch PSC của R3 đổ RC vào Line 1 ($y\_b = 1$) hay Line 2 ($y\_b = 0$). Trong DRL: baked vào action space — "PSC on R3 → L1" vs. "PSC on R3 → L2" → **17 actions** tổng (xem §9). Trong "fixed R3" mode: $y\_b = 0$ cho tất cả.

\---

## **9. Solution Methods**

### **9.1 Tổng Quan — 4 Reactive Strategies + Deterministic Benchmark**

|Method|Loại|Vai trò|Tham gia thí nghiệm UPS?|
|-|-|-|-|
|**MILP**|Exact optimization|Deterministic benchmark — LP relaxation lower bound để verify chất lượng CP-SAT|**Không** — chỉ giải deterministic|
|**CP-SAT**|Constraint programming|Deterministic solver — xác lập performance ceiling lý thuyết (~$295k)|**Không** — re-solve ~2 phút, không practical|
|**Dispatching Heuristic**|Rule-based|**Core baseline** — đại diện operator practice. Null hypothesis.|**Có**|
|**Q-Learning**|Tabular RL|RL đơn giản nhất — không neural network. Test: learning có giúp hơn rules?|**Có**|
|**MaskedPPO**|End-to-end DRL|Deep RL — neural network học mọi thứ từ scratch. Action masking feasibility.|**Có**|
|**RL-Hyper-Heuristic**|Tool-based DRL|RL agent chọn từ tập simple heuristic tools. Domain knowledge + RL.|**Có**|

**MILP + CP-SAT benchmark:** Cả hai giải cùng mô hình toán deterministic. MILP cung cấp LP relaxation lower bound → verify CP-SAT quality. CP-SAT xác lập theoretical ceiling (~$295k). Không tham gia reactive experiments.

> **Ghi chú về CP-SAT reactive (đã loại bỏ):** CP-SAT event-triggered re-solve đã được implement và test, nhưng loại khỏi reactive comparison vì re-solve time (~2 phút trên i3-9100F) không phù hợp real-time. Code vẫn tồn tại trong codebase.

### **9.2 Bốn Chiến Lược Reactive — Chi Tiết**

#### **Strategy 1: Dispatching Heuristic (Core Baseline)**

Rule-based, myopic, zero computation. Đại diện cho cách operator hiện tại ra quyết định — không có lookahead, không học từ experience. Đây là **null hypothesis**: performance khi không có intelligence nào.

**Decision function — gọi mỗi khi roaster $r$ trở thành IDLE tại time $t$:**

```
DISPATCH(r, t, state):

STEP 1 — CÓ NÊN RANG MTO KHÔNG?
  eligible\\\_mto = \\\[j for j in MTO\\\_jobs if mto\\\_remaining\\\[j] > 0 AND sku(j) ∈ eligible\\\_skus\\\[r]]
  
  If eligible\\\_mto không rỗng:
    setup = 5 if last\\\_sku\\\[r] ≠ sku(j) else 0
    
    # Tính urgency: tổng thời gian MTO còn cần / thời gian còn lại đến due date
    total\\\_mto\\\_time = Σ(mto\\\_remaining\\\[j] × 15 + setup\\\_for\\\_j) for j in eligible\\\_mto
    time\\\_left = 240 - t
    urgency = total\\\_mto\\\_time / max(time\\\_left, 1)
    
    If urgency > 0.7:   ← threshold: khi >70% thời gian còn lại cần cho MTO
      → Chọn MTO job (Step 1a)
      → Đi Step 3
    Else:
      → Chưa urgent, kiểm tra PSC (Step 2)

STEP 1a — CHỌN MTO JOB NÀO? (khi nhiều job eligible)
  → Priority: job có NHIỀU remaining batches hơn
  → Tie-break: Busta > NDG (Busta constrained hơn — chỉ R2 rang được)
  
  Ví dụ: R2 IDLE, j1(NDG, 2 remaining) vs j2(Busta, 2 remaining) → chọn Busta (tie-break)
  Ví dụ: R2 IDLE, j1(NDG, 3 remaining) vs j2(Busta, 1 remaining) → chọn NDG (3 > 1)

STEP 2 — RANG PSC
  # Kiểm tra overflow: batch này hoàn thành thì RC có tràn không?
  l\\\_out = output\\\_line(r)    # R1,R2 → L1. R4,R5 → L2. R3 → xem Step 2a.
  setup = 5 if last\\\_sku\\\[r] ≠ PSC else 0
  completion = t + setup + 15
  
  # Project RC tại thời điểm completion
  future\\\_consumption = |{τ ∈ E\\\[l\\\_out] : t < τ ≤ completion}|
  future\\\_completions = (batches đang rang sẽ hoàn thành trước completion)
  projected\\\_rc = rc\\\_stock\\\[l\\\_out] + future\\\_completions - future\\\_consumption + 1
  
  If projected\\\_rc > 40:
    → WAIT (RC sẽ tràn, đợi consumption giảm bớt)
  Else:
    → START PSC trên r (đi Step 3)

STEP 2a — R3 ROUTING (chỉ khi r = R3)
  If rc\\\_stock\\\[L1] ≤ rc\\\_stock\\\[L2]:
    → Route R3 output → L1 (y\\\_b = 1)   # Line 1 cần hỗ trợ hơn
  Else:
    → Route R3 output → L2 (y\\\_b = 0)
  # Sau đó kiểm tra overflow trên l\\\_out đã chọn (như Step 2)
  # Nếu l\\\_out overflow → thử line còn lại
  # Nếu cả 2 line overflow → WAIT

STEP 3 — PIPELINE \\\& DOWNTIME CHECK
  l\\\_pipe = pipe(r)
  setup = 5 if last\\\_sku\\\[r] ≠ sku(action) else 0
  
  If setup > 0:
    → Vào SETUP state (5 phút). Pipeline không bị chiếm.
    → Khi SETUP xong → IDLE → gọi lại DISPATCH()
  Else:
    If pipeline\\\_busy\\\[l\\\_pipe] > 0:
      → WAIT (pipeline đang bận, thử lại slot sau)
    Else:
      → START batch ngay lập tức
  
  # Downtime check
  start\\\_time = t + setup
  If \\\[start\\\_time, start\\\_time + 14] ∩ D\\\[r] ≠ ∅:
    → WAIT (không thể hoàn thành trước downtime)
```

**Điểm yếu chiến lược của Dispatching (so với CP-SAT/DRL):**

* **Myopic R3 routing:** Route theo stock hiện tại, không dự đoán stock tương lai
* **MTO timing:** Defer MTO cho đến khi "urgent" (>70%) → gây setup time cluster cuối deadline
* **Không coordinate pipeline:** Nếu R1, R2 cùng IDLE, xử lý tuần tự → roaster thứ 2 luôn đợi 3 phút
* **Không học từ UPS:** Sau UPS, áp dụng cùng rules — không thích ứng

#### **Strategy 2: Q-Learning (Tabular RL)**

Tabular RL — đơn giản nhất trong các phương pháp học. Không neural network, không gradient. Q-table lưu Q(state, action) cho mỗi cặp (state, action) đã visit.

**Kết quả empirical (non-UPS):** 1.4M episodes, 8.3h training → avg profit $296,270 ≈ CP-SAT $295k. Q-table chỉ 9,233 entries. Restock L2-PSC là argmax trong 34% states → **fully learned restock**.

**Tại sao Q-Learning work ở đây:**
* Không gradient death (không neural network, không shared backbone)
* Q(state, action) update độc lập per entry — không entry nào giết entry nào
* State discretization cẩn thận: 10 fields × vài giá trị = tractable tabular
* 1.4M episodes = mỗi (state, action) pair được visit hàng trăm lần

**Câu hỏi mở:** Q-Learning perform thế nào dưới UPS? State discretization có capture đủ UPS dynamics?

#### **Strategy 3: MaskedPPO (End-to-end DRL)**

End-to-end Deep RL — neural network (256,256 shared backbone) học mọi thứ từ raw observations (33 features) và actions (21 discrete). MaskablePPO (sb3-contrib) với action masking đảm bảo feasibility.

**Observation vector (33 features):** 27 base (time, 5× roaster status/timer/last\_sku, 2× RC, MTO remaining, 2× pipeline, 4× GC silo, restock status/timer) + 6 context one-hot (which roaster deciding).

**Action space (21 actions):** 9 batch start (bao gồm R3 routing baked in: action 2 = R3→L1, action 3 = R3→L2) + 4 restock + 7 reserved + WAIT.

**Reward:** Per-step incremental profit ($) + RC maintenance bonus + danger zone penalty.

**Vấn đề gradient death:** Sau 18 training cycles, best eval = $127,900 (C12, seed 300). Root cause: shared policy/value backbone → value network converge → advantage ≈ 0 → policy gradient chết → policy đóng băng. Architectural fix đề xuất: separate policy/value networks.

#### **Strategy 4: RL-Hyper-Heuristic (Tool-Based DRL)** 🆕

RL-based Hyper-Heuristic (RL-HH) — agent không trực tiếp chọn raw actions mà chọn từ tập **simple heuristic tools** (mỗi tool = 3-5 dòng logic). Agent HỌC cách **phối hợp** tools theo context. Tools biết HOW (domain knowledge), agent học WHEN (timing).

**Literature:** 86 publications 2020–2024 (Li et al. 2024, PeerJ Comput. Sci.). Key refs: Karimi-Mamaghan et al. (2023, EJOR), Panzer et al. (2024, IJPR), PetriRL (2026).

**6 Tools:**
* Tool 0: Rang PSC, output cho line có RC thấp nhất
* Tool 1: Restock GC silo thấp nhất nếu pipeline free
* Tool 2: Rang MTO nếu urgency > threshold
* Tool 3: Rang PSC trên roaster cùng SKU (tránh setup)
* Tool 4: R3 route sang line có RC thấp hơn
* Tool 5: WAIT

**Meta-agent:** Q-Learning (tabular) — action space chỉ 6 tools. State discretization meta-level (~432 states). Q-table ~2,600 entries. Không gradient death. Training ~1-2h.

**Tại sao approach này mạnh:**
* Action space 6 (vs 21) → tabular RL đủ
* Domain knowledge embedded trong tools → agent chỉ cần học timing
* Interpretable: "agent chọn Tool 1 (restock) vì GC thấp" = giải thích được
* Không gradient death vì không neural network

**Paradigm contrast — 4 Reactive Methods:**

|Dimension|Dispatching|Q-Learning|MaskedPPO|RL-HH|
|-|-|-|-|-|
|Memory|Không|Stateful (Q-table)|Stateful (neural net)|Stateful (meta Q-table)|
|Optimality|N/A (rules)|Approximate|Approximate|Approximate|
|Speed|<0.01ms|<0.01ms|~1ms|<0.1ms|
|Adaptability|Không (fixed rules)|Có (Q-table update)|Có (neural net)|Có (meta Q-table + tools)|
|Transparency|Hoàn toàn (rules)|Partial (Q-values)|Black box|Cao (tool name = explanation)|

Câu hỏi nghiên cứu: dưới disruption intensity nào, mỗi method chiếm ưu thế? Thêm complexity (rules → tabular → deep → structured) có thêm value không?

### **9.3 Simulation Loop**

Cả ba reactive strategies chạy trong cùng **simulation engine** — đảm bảo công bằng. Chi tiết đầy đủ: xem `event\_simulation\_logic\_complete.md`.

```
Main Loop — for t = 0 to 479:
  Phase 1: Process UPS events (cancel batch, set roaster DOWN, trigger re-plan)
  Phase 2: Advance roaster timers (batch complete → RC +1, → IDLE = decision point)
  Phase 3: Advance pipeline timers (consume done → pipeline free)
  Phase 4: Process consumption events (RC −1, track stockout)
  Phase 5: Process decision points (strategy decides next action per IDLE roaster)
```

**Phase order matters:** Batch completion (Phase 2) before consumption (Phase 4) → batch output "arrives just in time" to prevent stockout if both occur at same slot.

### **9.4 State Vector (dùng chung)**

|Component|Type|Range|Dùng bởi|
|-|-|-|-|
|$t\_0$|int|\[0, 479]|Tất cả|
|$\\text{status}\_r$|enum|{IDLE, RUNNING, DOWN, SETUP}|Tất cả|
|$\\text{remaining}\_r$|int|\[0, 30]|Tất cả|
|$\\text{last\_sku}\_r$|SKU|{PSC, NDG, Busta}|Tất cả (start = PSC)|
|$B\_l(t\_0)$|int|\[0, 40]|Tất cả|
|$G\_{l,k}(t\_0)$|int|\[0, $\overline{G}\_{l,k}$]|Tất cả|
|$\\text{MTO\_remaining}\_j$|int|\[0, $n\_j$]|Tất cả|
|$\\text{pipeline\_busy}\_l$|int|\[0, 15]|Tất cả (max 15 during restock)|
|$\\text{restock\_station\_busy}$|bool|{0, 1}|Tất cả|
|$\\text{restock\_timer}\_l$|int|\[0, 15]|Tất cả|

\---

## **10. Experimental Design**

### **10.1 Thí Nghiệm 1: MILP vs. CP-SAT Deterministic Benchmark**

Cùng model, cùng instance, hai solver khác nhau. Mục đích: verify CP-SAT solution quality.

|Parameter|Value|
|-|-|
|Instances|100 replications × 2 R3 modes = 200 runs|
|UPS|$\\lambda = 0$ (không có disruption)|
|Metrics|Objective value (profit $), solve time, LP relaxation gap|
|Expected finding|CP-SAT achieves same/near-optimal profit as MILP, faster solve time|

### **10.2 Thí Nghiệm 2: Reactive Strategy Comparison (Primary)**

|Factor|Levels|Count|
|-|-|-|
|UPS rate $\\lambda$|0, 1, 2, 3, 5 events/shift|5|
|UPS duration $\\mu$|10, 20, 30 min (mean)|3|
|Strategy|Dispatching / CP-SAT / DRL|3|
|R3 routing|Fixed (Line 2 only) / Flexible|2|

Full factorial: 5 × 3 × 3 × 2 = **90 cells** × 100 replications = **9,000 runs**.

**Paired comparison:** Cùng 100 UPS realizations cho cả 3 strategies trong mỗi cell. Khác biệt KPI **chỉ** đến từ strategy.

### **10.3 KPIs**

|KPI|Định nghĩa|Đơn vị|
|-|-|-|
|**Total profit**|Revenue − all costs (primary metric)|$|
|**PSC throughput**|Tổng PSC batch hoàn thành|batches|
|**Stockout count**|Consumption events với $B\_l < 0$ (strictly negative, demand unmet). **Penalized in objective.**|events|
|**Stockout duration**|Tổng phút $B\_l \leq 0$ (includes zero — line stalled). **KPI only, not in objective.**|minutes|
|**MTO tardiness**|$\sum\_j \text{tard}\_j$|minutes|
|**Re-solve count**|Số lần CP-SAT re-solve|count|
|**Compute time/decision**|Wall-clock per re-solve / inference|seconds|

**Lưu ý quan trọng — Stockout count vs. Stockout duration:** Objective function / DRL reward **tối ưu hóa stockout count** — số consumption events với $B\_l < 0$ (strictly negative). $B\_l = 0$ nghĩa là demand cuối cùng đã được phục vụ nhưng stock trống — đây **không phải** stockout event. Stockout duration (tổng phút $B\_l \leq 0$, bao gồm zero) là **KPI báo cáo riêng** cho phân tích vận hành. Cả hai đều được báo cáo, nhưng chỉ stockout count nằm trong objective.

\---

## **11. Assumptions**

|ID|Assumption|Chi tiết|Risk nếu violated|
|-|-|-|-|
|A1|Roasting time phụ thuộc SKU|PSC = 15 min, NDG = 17 min, Busta = 18 min|Thực tế có thể biến thiên ±1 min trong cùng SKU do điều kiện hạt|
|A2|Setup time đồng nhất 5 phút|Mọi cặp SKU|Một số transition nhanh/chậm hơn|
|A3|GC silo hữu hạn với restock cố định|L1: PSC(40), NDG(10), Busta(10). L2: PSC(40). Restock = 5 batch / 15 min|Thực tế restock time có thể biến thiên; silo capacity là xấp xỉ|
|A4|PSC consumption rate cố định|Không đổi suốt ca|Line speed adjustment không capture|
|A5|UPS parameters synthetic|Literature MTBF/MTTR|Relative performance, not absolute|
|A6|UPS ảnh hưởng đúng 1 roaster|Không correlated failures|Underestimate nếu mất điện|
|A7|UPS hủy batch hoàn toàn|Không pause/resume|Slightly pessimistic|
|A8|Không shelf life / FIFO|Chấp nhận 8 giờ|Multi-shift cần extend|
|A9|RC aggregate counter|Không silo-level|1 PSC SKU → no mixing|
|A10|Initial last\_sku = PSC|Setup cần cho MTO batch đầu ca|Hợp lý — factory default PSC|
|A11|Cost values are proxy|Không phải audited factory financials|Ratios matter, not absolutes|
|A12|MILP deterministic only|Không reactive MILP re-solve|MILP quá chậm cho real-time|
|A13|Shared restock station|Chỉ 1 restock tại 1 thời điểm toàn nhà máy|Nếu có nhiều forklift, constraint relaxed|
|A14|Restock size cố định 5 batch|Không partial restock|Thực tế có thể nạp lượng khác nhau|

\---

## **12. Out of Scope**

* GC silo SKU purity tracking (cross-contamination)
* GC dump operations (discarding GC from silo)
* Variable restock quantities (fixed at 5 batches per restock)
* RC individual silo-level tracking (fill rules, draw rules)
* PSC-PSC SKU switching within a shift
* 4-hour changeover
* Bill of Materials tracking (GC quantities per batch)
* Shelf life và FIFO
* Chất lượng rang / roast profile optimization
* Nhân công / labor constraints
* Stochastic demand (consumption rate là deterministic)
* Multi-shift planning / cross-shift carryover
* Setup time matrix (dùng giá trị đồng nhất 5 phút)
* Proactive robust scheduling
* Rolling horizon decomposition
* Metaheuristic approaches (ALNS — documented as future work for expandability)

