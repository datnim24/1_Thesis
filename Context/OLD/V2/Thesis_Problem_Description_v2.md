# **Problem Statement — Reactive Batch Roasting Scheduling Under Unplanned Stoppages at Nestlé Trị An**

---

## **1. Tổng Quan Bài Toán**

Bài toán đặt ra là xây dựng và đánh giá một hệ thống **reactive scheduling** (lập lịch phản ứng) cho quá trình rang cà phê theo batch tại nhà máy Nestlé Trị An, trong phạm vi một ca làm việc kéo dài 8 giờ, được discretize thành **480 time slot, mỗi slot = 1 phút** (slot 0 đến slot 479). Đây là bài toán lập lịch đa máy, đa sản phẩm trên 5 unrelated parallel roasters thuộc 2 production lines.

Hệ thống phải đồng thời giải quyết hai bài toán con lồng vào nhau:

* **Bài toán 1 — MTO (Make-to-Order):** Đáp ứng các đơn hàng NDG và Busta với due date mềm tại nửa ca (slot 240). Số lượng batch MTO nhỏ (3–5 batch) nhưng bắt buộc hoàn thành, và chỉ chạy được trên một số roaster nhất định (R1/R2 cho NDG, R2 cho Busta), tạo ra **resource contention** trực tiếp với production PSC trên cùng roaster.

* **Bài toán 2 — MTS (Make-to-Stock):** Duy trì tồn kho RC (Roasted Coffee) cho PSC ở mức ổn định bằng cách liên tục lập lịch các batch PSC trên tất cả 5 roasters, đáp ứng consumption rate cố định của dây chuyền đóng gói PSC. Nếu RC stock về 0, dây chuyền đóng gói dừng (**stockout**). Nếu RC stock đầy, roaster không có chỗ đổ output và phải dừng (**overflow**). Cả hai đều là failure mode phải phòng tránh.

Hai bài toán con này lồng vào nhau vì MTO batch chiếm roaster và pipeline — mỗi phút R1 hoặc R2 dành cho NDG/Busta là một phút không rang PSC, trực tiếp làm giảm RC supply rate. Scheduler phải cân bằng: đáp ứng MTO đúng hạn vs. duy trì RC buffer cho PSC.

**Ràng buộc vật lý cốt lõi** là **shared GC pipeline per line** — mỗi line có một đường ống dùng chung mà tất cả roaster trên line đó phải cạnh tranh quyền truy cập. Đường ống phục vụ thao tác **consume** (lấy green coffee cho batch), kéo dài 3 phút mỗi lần, và **không cho phép overlap**. Trên Line 2 với 3 roasters, pipeline utilization đạt **60%**, chỉ còn 6 phút slack per cycle — đủ chặt để bất kỳ disruption nào cũng gây cascading delay sang các roaster khác.

**Yếu tố stochastic chính** là **Unplanned Stoppages (UPS)** — sự cố thiết bị ngoài kế hoạch xảy ra theo phân phối xác suất, không biết trước tại thời điểm lập lịch ban đầu. Khi UPS xảy ra giữa ca, batch đang rang trên roaster bị ảnh hưởng **bị hủy hoàn toàn** (GC đã consume mất, phải restart từ đầu nếu muốn), và hệ thống phải re-schedule phần còn lại của ca dựa trên trạng thái hiện tại. Đây là thách thức trung tâm của luận văn: **làm thế nào để reactive scheduling strategy tốt nhất phục hồi throughput và duy trì RC stock sau disruption?**

Luận văn so sánh ba chiến lược reactive scheduling — dispatching heuristic (baseline), CP-SAT re-optimization (optimization-based), và Deep Reinforcement Learning policy (learning-based) — trên một thiết kế thí nghiệm factorial có kiểm soát, thay đổi cường độ disruption để xác định điều kiện mà mỗi chiến lược chiếm ưu thế.

Mục tiêu tối ưu là **maximize throughput** (tổng số PSC batch hoàn thành trong ca) trừ đi penalty cho MTO tardiness, subject to hard constraints cho RC inventory bounds (không stockout, không overflow).

---

## **2. Cấu Hình Hệ Thống Vật Lý**

### **2.1 Hai Line Sản Xuất**

Nhà máy có hai production lines hoạt động song song.

**Line 1** bao gồm **Roaster 1** và **Roaster 2**, với RC buffer riêng (aggregate batch counter).

**Line 2** bao gồm **Roaster 3**, **Roaster 4** và **Roaster 5**, với RC buffer riêng.

Hai line **độc lập hoàn toàn** về RC inventory và pipeline — pipeline Line 1 chỉ phục vụ R1, R2; pipeline Line 2 phục vụ R3, R4, R5. Ngoại trừ **Roaster 3** — điểm kết nối chéo duy nhất: R3 luôn consume GC từ Line 2 pipeline, nhưng RC output có thể đổ vào RC stock của Line 1 hoặc Line 2 tùy quyết định scheduler (xem §2.2).

Hệ quả: nếu không có R3 cross-line routing, hai line có thể được giải quyết **hoàn toàn độc lập** — hai bài toán scheduling riêng biệt nhỏ hơn. Với R3 routing, hai line **coupled** thông qua R3, tạo ra bài toán scheduling chung không thể tách rời.

### **2.2 Năm Roasters — Eligibility, Pipeline Mapping, và RC Output**

Mỗi roaster xử lý tối đa **1 batch** tại một thời điểm. Thời gian rang cố định **15 phút** cho mọi SKU, mọi roaster (xem Assumption A1).

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
* RC output: **Line 1 hoặc Line 2** — đây là **decision variable** ($y_b \in \{0,1\}$), scheduler quyết định cho từng batch
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

| Roaster | Line | SKU eligible | Pipeline consume | RC output | Đặc điểm |
|---------|------|-------------|------------------|-----------|-----------|
| R1 | 1 | PSC, NDG | Line 1 | Line 1 (fixed) | MTO capable, shared capacity |
| R2 | 1 | PSC, NDG, Busta | Line 1 | Line 1 (fixed) | Most flexible, Busta-only capable |
| R3 | 2 | PSC | Line 2 (always) | **Line 1 or 2 (decision)** | Cross-line bridge |
| R4 | 2 | PSC | Line 2 | Line 2 (fixed) | Dedicated PSC |
| R5 | 2 | PSC | Line 2 | Line 2 (fixed) | Dedicated PSC |

### **2.3 Sequence-Dependent Setup Time — Ràng Buộc Cứng**

Khi hai batch **liên tiếp trên cùng một roaster** có **SKU khác nhau**, bắt buộc phải có **5 phút (5 time slot) setup time** giữa thời điểm kết thúc batch trước và thời điểm bắt đầu batch sau.

**Quy tắc chi tiết:**

* Setup áp dụng cho **mọi loại transition SKU** mà không phân biệt mức độ khác nhau: PSC → NDG, NDG → PSC, PSC → Busta, Busta → PSC, NDG → Busta, Busta → NDG — **tất cả đều tốn đúng 5 phút** (Assumption A2).
* Nếu hai batch liên tiếp **cùng SKU**, không có setup time — batch sau có thể bắt đầu ngay (chỉ cần pipeline rảnh).
* Trong 5 phút setup, roaster **không làm gì** — không rang, không consume GC. Roaster ở trạng thái SETUP.
* Đường ống GC **không bị chiếm** bởi setup time — pipeline tự do phục vụ consume cho roaster khác trên cùng line.
* Setup time là **hard constraint** — không thể rút ngắn, không thể bỏ qua.

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

**Hệ quả chiến lược:** Scheduler bị khuyến khích tự nhiên **nhóm tất cả batch cùng SKU liên tiếp** trên cùng roaster. Mỗi lần xen kẽ SKU tốn 5 phút idle — chi phí này trực tiếp giảm throughput mà không cần penalty riêng. Ví dụ: nếu R1 cần rang 3 batch NDG và 10 batch PSC, lịch tối ưu là [3× NDG] → [setup 5 min] → [10× PSC], chỉ tốn 1 lần setup = 5 phút. Nếu xen kẽ NDG-PSC-NDG-PSC-NDG-PSC..., tốn 5 lần setup = 25 phút mất trắng.

### **2.4 Shared GC Pipeline — Ràng Buộc Vật Lý Cốt Lõi**

#### **2.4.1 Cấu trúc pipeline**

Mỗi line có **một đường ống GC (Green Coffee) duy nhất** kết nối nguồn GC với tất cả roaster trên line đó:

* **Pipeline Line 1:** phục vụ R1 và R2
* **Pipeline Line 2:** phục vụ R3, R4 và R5

Hai pipeline hoạt động **hoàn toàn độc lập** — Line 1 pipeline và Line 2 pipeline không ảnh hưởng lẫn nhau.

#### **2.4.2 Thao tác trên pipeline: chỉ có Consume**

Pipeline chỉ thực hiện một loại thao tác duy nhất: **consume** — lấy green coffee beans từ nguồn cung cấp vào roaster khi batch bắt đầu. GC supply được coi là **unlimited** (luôn có đủ GC, không theo dõi inventory GC, không có replenish hay dump).

Mỗi thao tác consume **kéo dài 3 phút** (3 time slot), bất kể loại SKU hay lượng GC.

#### **2.4.3 Mutual Exclusion — NoOverlap Constraint**

**Ràng buộc cốt lõi:** Tại bất kỳ thời điểm nào, pipeline của mỗi line chỉ phục vụ **tối đa 1 consume operation**. Không có exception.

Nếu pipeline đang bận phục vụ roaster A (đang trong 3 phút consume), roaster B phải **đợi** cho đến khi pipeline rảnh — ngay cả khi roaster B đã sẵn sàng (idle, không cần setup). Thời gian đợi pipeline gây ra **idle time ngoài kiểm soát** của roaster — đây là nguồn delay chính trong hệ thống.

#### **2.4.4 Consume Timing — Song Song Với Đầu Quá Trình Rang**

Thời điểm batch bắt đầu rang **đồng thời** là thời điểm consume bắt đầu. Pipeline bận trong **3 phút đầu** của batch, trong khi roaster bận trong **toàn bộ 15 phút** của batch. Hai khoảng thời gian này **overlap** — consume là hoạt động song song với đầu quá trình rang, **không phải** giai đoạn tuần tự trước khi rang.

```
Timeline chi tiết cho 1 batch bắt đầu tại t=0:

Pipeline:  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
            t=0  t=1  t=2  (3 slots bận)

Roaster:   ████████████████████████████████████████████████████████████
            t=0  t=1  t=2  t=3  ...  t=13  t=14  (15 slots bận)

           |---consume (3 min)---|
           |--------------------roasting (15 min)--------------------|
           t=0                  t=3                                 t=15

Pipeline trở lại FREE tại t=3 → roaster khác trên cùng line có thể bắt đầu consume tại t=3.
Roaster trở lại IDLE tại t=15 → batch hoàn thành, RC stock +1.
```

Điều này có nghĩa: ngay khi pipeline rảnh (sau 3 phút), roaster khác có thể bắt đầu batch tiếp theo **trong khi** roaster hiện tại vẫn đang rang. Đây là lý do nhiều batch có thể **overlap trên các roaster khác nhau** miễn là pipeline không conflict.

#### **2.4.5 Pipeline Utilization Analysis — Tại Sao Line 2 Là Bottleneck**

**Line 1 (R1, R2) — 2 roasters:**

Trong điều kiện lý tưởng (không setup, không downtime), mỗi roaster cần pipeline 3 phút mỗi 15 phút:
* Pipeline demand: 2 roasters × 3 min = **6 min per 15-min cycle**
* Pipeline capacity: 15 min
* **Utilization: 6/15 = 40%**
* Slack: **9 min** — rộng rãi. Hai roaster hiếm khi conflict.

Ví dụ lịch trình không conflict:
```
Pipeline Line 1:
R1: ███...............███...............███...............
R2: ...███...............███...............███............
    t=0  3  6  9  12 15 18 21 24 27 30 33 36 39 42 45

R1 consume: [0,2], [15,17], [30,32], ...
R2 consume: [3,5], [18,20], [33,35], ...
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

R3 consume: [0,2], [15,17], [30,32], ...
R4 consume: [3,5], [18,20], [33,35], ...
R5 consume: [6,8], [21,23], [36,38], ...
→ Không overlap. Nhưng nếu BẤT KỲ roaster nào bị delay 1 slot,
   consume sẽ chen vào slot của roaster tiếp theo → cascade.
```

**Cascading delay scenario trên Line 2:**

Giả sử R3 bị delay 2 phút (do setup hoặc UPS recovery):
```
Original:     R3:[0,2]   R4:[3,5]   R5:[6,8]
After delay:  R3:[2,4]   → conflict với R4 tại slot 3,4
              R4 phải shift: R4:[5,7]   → conflict với R5 tại slot 6,7  
              R5 phải shift: R5:[8,10]
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
    t=47: R1 bắt đầu PSC batch. Consume: [47,49]. Roast: [47,61].
    
    Total idle cho R1: t=40 → t=47 = 7 phút (5 setup + 2 pipeline wait).
```

### **2.5 RC Inventory — Aggregate Batch Counter Per Line**

#### **2.5.1 Cách theo dõi inventory**

RC (Roasted Coffee) inventory được theo dõi dưới dạng **một số nguyên** cho mỗi line, tính bằng **số batch** (không phải kg). Mỗi line có một bộ đếm RC stock duy nhất — không phân biệt từng RC silo riêng lẻ.

Ký hiệu: $B_l(t)$ = RC stock trên line $l$ tại time slot $t$ (đơn vị: batch).

#### **2.5.2 RC stock thay đổi bởi hai sự kiện**

**RC tăng (+1) khi:** Một batch **PSC** hoàn thành rang trên roaster có RC output thuộc line $l$. Batch hoàn thành tại thời điểm $t = s_b + 15$ (start time + processing time). Tại thời điểm đó, $B_l$ tăng 1 đơn vị.

Lưu ý: chỉ **PSC batch** mới tăng RC stock. NDG và Busta batch **không đổ vào RC** — sản phẩm MTO được giao thẳng cho khách, không qua inventory. Do đó, mỗi phút roaster dành cho MTO là một phút **không bổ sung RC stock** cho line đó.

**RC giảm (−1) khi:** PSC consumption event xảy ra theo lịch cố định (xem §3 PSC Consumption). Mỗi event tại thời điểm $\tau \in \mathcal{E}_l$ giảm $B_l$ đi 1 đơn vị.

#### **2.5.3 Inventory balance equation**

$$B_l(t) = B^0_l + \underbrace{\sum_{\substack{b \in \mathcal{B}^{PSC}:\\ \text{out}(b) = l,\; e_b \leq t}} 1}_{\text{PSC batches completed on line } l \text{ by time } t} \;\;-\;\; \underbrace{\left|\left\{ \tau \in \mathcal{E}_l : \tau \leq t \right\}\right|}_{\text{consumption events on line } l \text{ by time } t}$$

Trong đó $B^0_l$ là RC stock ban đầu (initial condition, input).

#### **2.5.4 Inventory bounds — hard constraints**

**Stockout — $B_l(t) < 0$:** Nghĩa là consumption event xảy ra nhưng không có RC batch nào để lấy. Dây chuyền PSC packaging **dừng** — đây là failure mode chính cần phòng tránh.

* Trong **deterministic mode** (lập lịch ban đầu, không có UPS): $B_l(t) \geq 0$ là **hard constraint**. Model phải tìm schedule sao cho RC không bao giờ về 0 khi có consumption event.
* Trong **reactive mode** (sau UPS, re-solve): stockout có thể **unavoidable** (ví dụ: 2 roasters trên Line 1 cùng down 30 phút, consumption vẫn tiếp tục). Lúc này $B_l(t) \geq 0$ chuyển thành **soft constraint** với heavy penalty $w^{stock}$ — model tìm schedule **minimizes stockout duration** thay vì tuyên bố infeasible.

**Overflow — $B_l(t) > \overline{B}_l$:** Nghĩa là roaster hoàn thành PSC batch nhưng RC stock đã đầy — không có chỗ chứa output. Roaster **phải dừng** — không thể xả RC. Đây là **hard constraint trong mọi mode** vì overflow là physical impossibility (silo đầy vật lý, không thể ép thêm).

$\overline{B}_l$ = maximum RC buffer capacity per line, tính bằng batch units = total physical RC silo capacity (kg) / batch output size (kg). *Giá trị cụ thể cần tính từ dữ liệu nhà máy (TBD).*

#### **2.5.5 Không có SKU mixing issue**

Mỗi ca chỉ sản xuất **1 PSC SKU duy nhất**. Không có PSC-PSC SKU switch. RC inventory luôn chứa single-SKU. Do đó, không cần theo dõi SKU assignment trong RC, không cần switching logic, không cần threshold conditions.

#### **2.5.6 Ví dụ RC inventory trajectory**

Giả sử Line 1, $B^0_{L1} = 10$ batch, consumption rate $\rho = 5$ phút/batch (1 batch consumed mỗi 5 phút), 2 roasters (R1, R2) cả hai đang rang PSC liên tục:

```
Thời gian  | Events                          | B_L1
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

---

## **3. PSC Consumption**

### **3.1 Consumption Rate**

PSC tiêu thụ RC theo lịch **deterministic** với tốc độ cố định, đặc trưng cho từng PSC SKU và từng line.

**Consumption rate** $\rho_l$: Mỗi line tiêu thụ 1 batch RC mỗi $\rho_l$ phút. Giá trị $\rho_l$ là **input** của bài toán, được xác định bởi PSC SKU đang chạy trên line đó.

Ví dụ: $\rho_l = 5.1$ phút nghĩa là dây chuyền PSC packaging của line $l$ lấy 1 batch RC từ stock khoảng mỗi 5.1 phút.

### **3.2 Consumption Schedule — Precomputed Input**

Từ rate $\rho_l$, tính trước **danh sách thời điểm consumption events** trước khi bắt đầu solve:

$$\mathcal{E}_l = \left\{ \lfloor i \cdot \rho_l \rfloor \;\middle|\; i = 1, 2, \dots, \left\lfloor \frac{480}{\rho_l} \right\rfloor \right\}$$

**Ví dụ chi tiết:** $\rho_l = 5.1$ phút:

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

Mỗi event tại thời điểm $\tau \in \mathcal{E}_l$ giảm $B_l$ đi 1 đơn vị.

### **3.3 Tính chất quan trọng**

* Consumption rate **không đổi** trong suốt ca.
* **Chỉ 1 PSC SKU** per line per shift — không có PSC-PSC SKU switch, không có thay đổi rate giữa ca.
* Consumption xảy ra **bất kể** tình trạng scheduling — dây chuyền đóng gói chạy liên tục và không quan tâm roasters đang làm gì. Nếu RC stock = 0 khi consumption event đến, đây là **stockout**.
* Consumption schedule là **input cố định** — scheduler không thể thay đổi hay trì hoãn consumption. Scheduler chỉ có thể điều chỉnh **production schedule** (khi nào rang batch nào) để đảm bảo RC luôn đủ.

---

## **4. Sản Phẩm, Jobs và Batches**

### **4.1 NDG và Busta — Make-to-Order (MTO)**

NDG và Busta là sản phẩm sản xuất **theo đơn hàng**. Toàn bộ thông tin MTO đã biết từ đầu ca:

* **Loại SKU:** NDG hoặc Busta (hai SKU riêng biệt)
* **Số batch cần rang:** Thường tổng cộng **3–5 batch** cho cả NDG và Busta trong ca
* **Roaster eligibility:**
  * NDG: rang được trên **R1 hoặc R2** (scheduler chọn)
  * Busta: **chỉ R2** (không có roaster nào khác rang được)

**MTO Jobs:** Mỗi MTO order (ví dụ: "4 batch NDG") là 1 **job** gồm nhiều batch. Tất cả batch của cùng job có cùng SKU. Job hoàn thành khi batch cuối cùng của job hoàn thành.

**Due date — soft constraint:** Tất cả batch MTO phải hoàn thành trước **slot 240** (nửa ca = 4 giờ). Đây là **soft constraint** với penalty weight $w^{tard}$:

$$\text{tard}_j = \max\left(0,\;\; \max_{b \in \mathcal{B}_j} (s_b + 15) - 240\right)$$

Tardiness tính theo batch **cuối cùng** của job — nếu batch cuối hoàn thành tại slot 250, tardiness = 10 phút.

Lý do dùng soft constraint (không hard): trong tình huống cực đoan — ví dụ UPS trên R2 kéo dài 45 phút trong giờ đầu ca khi có Busta order (chỉ R2 rang được) — việc đáp ứng due date có thể physically impossible. Hard constraint sẽ khiến model báo infeasible và không đưa ra schedule nào. Soft constraint cho phép model tìm **schedule tốt nhất có thể** và báo cáo mức tardiness.

**RC output:** NDG và Busta **không đổ vào RC stock**. Sản phẩm MTO hoàn thành được giao thẳng, không qua inventory. MTO batch chỉ ảnh hưởng đến scheduling:
* Chiếm roaster (15 phút)
* Chiếm pipeline (3 phút)
* Có thể gây setup time (nếu roaster phải chuyển từ PSC → NDG hoặc ngược lại)

Nhưng **không ảnh hưởng đến RC balance** — RC stock chỉ tăng khi PSC batch hoàn thành.

### **4.2 PSC — Make-to-Stock (MTS)**

PSC là sản phẩm rang để **bổ sung RC inventory**. Không có ticket cố định — scheduler tự quyết định:

* **Bao nhiêu batch PSC** cần rang trong ca (decision variable — không biết trước)
* **Trên roaster nào** (tất cả 5 roasters đều eligible cho PSC)
* **Tại thời điểm nào** (phải cân bằng: rang đủ sớm để tránh stockout, không rang quá nhiều gây overflow)

Mỗi ca chỉ sản xuất **1 PSC SKU duy nhất** — không có PSC-PSC SKU switch.

PSC batch count là **decision variable** — model sử dụng **pool of optional batches** (pre-generated) và solver chọn bao nhiêu batch cần activate. Pool size = theoretical maximum = $\lfloor 480/15 \rfloor = 32$ per roaster × 5 roasters = 160 optional PSC batches. Trong thực tế, solver chỉ activate ~50–80 batch (phụ thuộc vào consumption rate và MTO demand).

### **4.3 Tương tác giữa MTO và MTS — Resource Contention**

Đây là tension trung tâm của bài toán:

* R1 và R2 phải **chia capacity** giữa MTO (bắt buộc, có deadline) và PSC (tự nguyện, nhưng cần để tránh stockout).
* Mỗi batch NDG/Busta trên R1/R2 = 15 phút roaster + 3 phút pipeline **không rang PSC** = Line 1 RC stock giảm.
* Nếu NDG/Busta order lớn (5 batch), R1/R2 dành ~75 phút + setup time cho MTO → Line 1 thiếu PSC supply → cần R3 route output về Line 1 để bù.
* Setup time khi chuyển NDG → PSC (5 phút) thêm chi phí cho mỗi lần xen kẽ.

Ví dụ cụ thể:
```
Ca có: 4 batch NDG (R1 hoặc R2) + 1 batch Busta (R2 only).
Consumption rate Line 1: 1 batch / 5 phút = 96 batches needed per shift.

R2 phải rang: 1 Busta + PSC còn lại
R1 phải rang: 4 NDG + PSC còn lại

Thời gian MTO trên Line 1: (4 + 1) × 15 min + 2 × 5 min setup = 85 phút
                                                    (NDG→PSC + Busta→PSC)
Thời gian còn lại cho PSC trên Line 1: (2 × 480) − 85 = 875 phút roaster-time
PSC batches from Line 1: 875 / 15 ≈ 58 batches
Consumption events: 96 batches needed
Gap: 96 − 58 = 38 batches phải đến từ R3 routing
```

---

## **5. Planned Downtime**

### **5.1 Định nghĩa**

Planned downtime là các khoảng thời gian đã lên lịch trước mà roaster không thể hoạt động. Thông tin này là **input cố định**, biết từ đầu ca, dưới dạng tập hợp time slots cho mỗi roaster:

$$\mathcal{D}_r \subset \mathcal{T} = \{0, 1, \dots, 479\} \quad \forall r \in \mathcal{R}$$

Downtime có thể khai báo:
* **Per roaster:** Chỉ R3 down trong [100, 130]
* **Per line:** R1 và R2 cùng down trong [200, 250]
* **Toàn nhà máy:** Tất cả R1-R5 down trong [300, 320]

### **5.2 Quy tắc chi tiết**

**Quy tắc 1 — Không được start batch nếu không thể hoàn thành trước downtime:**

Batch chỉ được start tại slot $t$ trên roaster $r$ nếu toàn bộ khoảng processing $[t, t+14]$ (15 slots) **không overlap** với $\mathcal{D}_r$.

$$[s_b, s_b + 14] \cap \mathcal{D}_r = \emptyset \quad \forall b : r_b = r$$

Nếu downtime bắt đầu tại slot 100, batch cuối cùng có thể start là tại slot 85 (kết thúc tại slot 99). Roaster phải idle từ slot 86 (nếu slot 85 đã bị chiếm) cho đến khi downtime kết thúc.

**Quy tắc 2 — Không có mid-batch pause/resume cho planned downtime:**

Scheduler biết trước downtime windows và **phải** lập lịch tránh chúng. Không có tình huống batch "bị cắt ngang" bởi planned downtime — đó là lỗi scheduling.

**Quy tắc 3 — Sau downtime kết thúc:**

Roaster trở lại trạng thái **IDLE** và có thể start batch mới bình thường. Không cần setup — **trừ khi** batch mới có SKU khác với batch cuối cùng trước downtime (thì phải setup 5 phút bình thường).

**Ví dụ:**
```
R4 planned downtime: [120, 180] (1 giờ, từ phút 120 đến phút 180)
R4 đang rang PSC.

Batch cuối trước downtime:
  → Phải start trước slot 106 (106 + 14 = 120, vừa kịp hoàn thành trước downtime)
  → Nếu batch trước kết thúc tại slot 105, có thể start batch mới tại 105
  → Nhưng 105 + 14 = 119 < 120 → OK
  → Nếu batch trước kết thúc tại slot 108: 108 + 14 = 122 > 120 → KHÔNG ĐƯỢC START
  → R4 phải idle từ 108 đến 180. Total idle: 72 phút (12 phút trước downtime + 60 phút downtime).

Sau downtime:
  → t=180: R4 IDLE. Batch tiếp theo là PSC (cùng SKU) → không cần setup.
  → t=180: R4 start PSC batch (nếu pipeline rảnh). Consume: [180,182]. Roast: [180,194].
```

---

## **6. Unplanned Stoppages (UPS) — Yếu Tố Stochastic Trung Tâm**

UPS là sự cố thiết bị ngoài kế hoạch, **không biết trước** tại thời điểm lập lịch ban đầu. UPS là **yếu tố stochastic trung tâm** của luận văn — toàn bộ reactive scheduling framework được xây dựng xung quanh việc xử lý UPS.

### **6.1 UPS Event Model**

Mỗi UPS event được đặc trưng bởi ba tham số:

* **Thời điểm xảy ra $t^{UPS}$:** Phân phối Exponential inter-arrival với rate $\lambda$ (expected number of UPS events per shift).
  * $\lambda = 0$: Không có UPS (deterministic baseline)
  * $\lambda = 1$: Trung bình 1 UPS per shift (nhẹ)
  * $\lambda = 3$: Trung bình 3 UPS per shift (trung bình)
  * $\lambda = 5$: Trung bình 5 UPS per shift (nặng)

* **Duration $d$:** Thời gian roaster DOWN, drawn from phân phối tham số hóa với mean $\mu$ phút.
  * $\mu = 10$ phút: Sửa nhanh (thay bộ phận nhỏ)
  * $\mu = 20$ phút: Sửa trung bình
  * $\mu = 30$ phút: Sửa lớn (thay bộ phận chính)

* **Affected roaster:** Random uniform trên 5 roasters. Mỗi UPS ảnh hưởng **đúng 1 roaster** (Assumption A6 — không model correlated failures).

**Lưu ý quan trọng:** UPS parameters là **synthetic** — dựa trên ước lượng MTBF/MTTR từ literature, **không calibrate** từ dữ liệu Nestlé Trị An cụ thể. Đây là limitation được acknowledge. Thiết kế thí nghiệm factorial trên grid ($\lambda$, $\mu$) bù đắp bằng cách bao phủ **dải rộng** disruption intensity — kết quả cho thấy **relative performance** giữa strategies, không phải absolute prediction cho nhà máy.

### **6.2 UPS Impact — Batch Bị Hủy Hoàn Toàn**

Khi UPS xảy ra tại thời điểm $t^{UPS}$ trên roaster $r$:

**Trường hợp 1: Roaster $r$ đang rang batch $b$ (RUNNING state):**

1. **Batch $b$ bị hủy hoàn toàn.** Quá trình rang dừng ngay lập tức. Batch không thể salvage, không có partial output. Cà phê rang dở bên trong roaster bị coi là **waste**.
2. **GC đã consume cho batch $b$ mất.** 3 phút pipeline đã dùng cho consume ở đầu batch là **sunk cost** — pipeline time đã trôi, không lấy lại được. (Trong model, GC unlimited nên mất GC không có cost trực tiếp, nhưng pipeline time là real cost.)
3. **RC output không được tính.** Batch chưa hoàn thành → RC stock **không tăng** → consumption vẫn tiếp tục → RC giảm → tăng stockout risk.
4. **Roaster $r$ chuyển sang DOWN state** trong $d$ phút. Trong suốt thời gian DOWN, roaster không thể làm gì.

**Trường hợp 2: Roaster $r$ đang IDLE hoặc SETUP:**

1. Không có batch nào bị hủy (không có batch đang chạy).
2. Roaster $r$ chuyển sang **DOWN state** trong $d$ phút.
3. Nếu đang SETUP: setup timer **reset** — khi roaster trở lại, nếu batch tiếp theo vẫn cần setup, phải setup lại từ đầu 5 phút.

**Trường hợp 3: Roaster $r$ đang DOWN (từ UPS trước):**

Nếu UPS xảy ra lần nữa trên roaster đã down → duration **cộng dồn** hoặc **replace** (tùy convention; đề xuất: lấy max(remaining_down, new_duration) — tức là nếu UPS mới dài hơn, extend thời gian down).

### **6.3 Sau Khi UPS Kết Thúc — Scheduler Decision**

Khi roaster $r$ trở lại service (DOWN → IDLE), đây là một **decision point** — scheduler phải quyết định:

* **Có restart batch $b$ (batch bị hủy) hay không?** Đây là **decision variable**, không phải automatic restart.
  * Nếu **restart**: cần consume lại (3 phút pipeline) + rang lại toàn bộ (15 phút). Đây là batch **mới hoàn toàn** — pipeline phải available, tốn thêm 3 phút pipeline time.
  * Nếu **skip**: roaster tự do chọn batch khác (có thể khác SKU → thêm 5 phút setup).

* **Considerations cho quyết định restart:**
  * Nếu batch $b$ là **MTO** (NDG/Busta): skip = tăng tardiness penalty. Nếu due date gần, restart gần như bắt buộc.
  * Nếu batch $b$ là **PSC**: skip = mất 1 batch throughput, nhưng có thể schedule batch PSC khác ở slot tốt hơn. Restart không nhất thiết tối ưu nếu pipeline đang busy.
  * **RC stock level tại thời điểm đó:** Nếu stock rất thấp, ưu tiên rang PSC ngay (bất kể restart hay batch mới). Nếu stock comfortable, có thể rang MTO trước.

### **6.4 UPS Cascading Effect — Ví Dụ Chi Tiết Trên Line 2**

Đây là ví dụ minh họa tại sao UPS đặc biệt disruptive trên Line 2 (60% pipeline utilization):

```
Trạng thái bình thường trước UPS:
  R3: batch hoàn thành tại t=100, consume tiếp theo tại t=100: [100,102]
  R4: batch hoàn thành tại t=101, consume tiếp theo tại t=103: [103,105]
  R5: batch hoàn thành tại t=103, consume tiếp theo tại t=106: [106,108]
  → Pipeline schedule tốt: R3→R4→R5 interleaved, 1 phút gap mỗi lần.

UPS trên R4 tại t=95, duration = 20 phút:
  1. t=95: R4 batch đang rang bị HỦY. R4 DOWN đến t=115.
  2. t=100: R3 hoàn thành batch, consume bình thường [100,102]. ✓
  3. t=103: R5 hoàn thành batch.
     → Lẽ ra R4 consume tại [103,105] nhưng R4 down.
     → R5 consume tại [103,105]. ✓ (R4 slot bỏ trống, R5 lấy)
  4. t=115: R4 trở lại. Scheduler quyết định restart batch PSC.
     → R4 cần consume. Pipeline status tại t=115?
     → R3 consume cuối: [115,117] (R3 có batch mới start tại t=115)
     → CONFLICT! R4 phải đợi đến t=118.
     → R4 consume: [118,120]. Batch start: t=118, done: t=132.
  5. Cascade:
     → R4 consume tại [118,120] → chen vào slot dự kiến của R5
     → R5 phải shift consume từ [118,...] sang [121,...]
     → Total delay: R4 mất 20 phút (UPS) + 3 phút (pipeline wait)
                     R5 mất 3 phút (pipeline cascade)
     → Throughput loss: 1 batch hủy (R4) + ~1-2 batches delay (cascade)
```

### **6.5 UPS vs. Planned Downtime — So Sánh Chi Tiết**

| Thuộc tính | Planned Downtime | Unplanned Stoppage (UPS) |
|-----------|-----------------|--------------------------|
| **Biết trước?** | Có — input cố định từ đầu ca | Không — random event theo phân phối xác suất |
| **Scheduler có thể phòng tránh?** | Có — lập lịch tránh trước (hard constraint) | Không — chỉ có thể react sau khi xảy ra |
| **Mid-batch behavior** | Batch không start nếu không kịp hoàn thành | Batch đang rang bị **hủy hoàn toàn**, GC mất |
| **Pipeline impact khi xảy ra** | Không (scheduler đã tránh) | Consume trước đó là sunk cost; re-consume cần 3 phút pipeline mới |
| **Sau khi kết thúc** | Roaster IDLE, start batch bình thường | Scheduler **quyết định** restart batch bị hủy hay chuyển sang batch khác |
| **RC inventory impact** | Predictable (scheduler biết trước idle time) | Unpredictable (mất 1 batch output + thêm idle time) |
| **Trong model** | Hard constraint: $[s_b, e_b) \cap \mathcal{D}_r = \emptyset$ | Event trigger: re-solve CP-SAT / RL action / dispatching rule |

---

## **7. Objective Function — Maximize Throughput**

### **7.1 Deterministic Mode (Predictive Schedule)**

Model tối đa hóa tổng throughput trừ tardiness penalty:

$$\text{Maximize:} \quad \underbrace{\sum_{b \in \mathcal{B}^{PSC}} a_b}_{\text{PSC throughput}} \;\;-\;\; w^{tard} \cdot \underbrace{\sum_{j \in \mathcal{J}^{MTO}} \text{tard}_j}_{\text{MTO tardiness}}$$

Trong đó:
* $a_b \in \{0,1\}$: binary activation — 1 nếu PSC batch $b$ được activate (scheduled), 0 nếu không.
* $\text{tard}_j = \max(0,\; e_{b_{last}^j} - 240)$: tardiness của MTO job $j$. Tính từ batch **cuối cùng** của job. Đơn vị: phút.
* $w^{tard}$: penalty weight per minute of MTO tardiness.

**Calibration $w^{tard}$:** Cần calibrate sao cho 1 phút MTO tardiness "tệ hơn" việc mất 1 PSC batch. Logic: 1 PSC batch bù đắp ~$\rho_l$ phút consumption (ví dụ 5.1 phút). Nếu $w^{tard} = 5$, thì 1 phút MTO late = penalty 5, tương đương mất 5 PSC batches. Giá trị đề xuất: $w^{tard} \in [5, 10]$.

MTO batch count là **fixed** (luôn active, $a_b = 1$ cho mọi $b \in \mathcal{B}^{MTO}$), nên maximizing tổng throughput reduces to **maximizing PSC batch activations** (subject to constraints).

### **7.2 Reactive Mode (Post-UPS Re-solve)**

Sau UPS, thêm stockout penalty vì stockout có thể unavoidable:

$$\text{Maximize:} \quad \sum_{b \in \mathcal{B}^{PSC}_{rem}} a_b \;\;-\;\; w^{tard} \cdot \sum_{j} \text{tard}_j \;\;-\;\; w^{stock} \cdot \sum_{l \in \mathcal{L}} \sum_{t \in \mathcal{T}_{rem}} \text{stockout}_{l,t}$$

Trong đó:
* $\mathcal{B}^{PSC}_{rem}$: remaining unstarted PSC batches (batches đã hoàn thành bị freeze, không đổi).
* $\text{stockout}_{l,t} = \max(0, -B_l(t))$: deficit tại line $l$, time $t$. Bằng 0 nếu stock ≥ 0.
* $w^{stock}$: penalty weight per stockout event-minute.

**RC inventory bounds trong reactive mode:**
* Overflow ($B_l(t) \leq \overline{B}_l$): **hard constraint** (physical impossibility — không relax được).
* Stockout ($B_l(t) \geq 0$): **soft constraint** (penalty $w^{stock}$). Model tìm schedule tối thiểu hóa stockout thay vì báo infeasible.

---

## **8. Decision Variables**

Model tự quyết định toàn bộ các biến sau tại mỗi lần solve (initial hoặc re-solve):

* **Batch activation (PSC only):** $a_b \in \{0,1\}$ — batch PSC nào cần activate từ pool of optional batches. MTO batches luôn active ($a_b = 1$).
* **Roaster assignment:** $r_b \in \mathcal{R}_b$ — mỗi batch chạy trên roaster nào, giới hạn bởi eligibility (NDG → R1/R2, Busta → R2, PSC → R1-R5).
* **Start time:** $s_b \in \mathcal{T}$ — mỗi batch bắt đầu tại time slot nào. Phải tôn trọng đồng thời:
  * Setup time 5 phút nếu đổi SKU so với batch trước trên cùng roaster
  * Pipeline availability (3 phút consume không overlap với batch khác trên cùng line)
  * Planned downtime clearance (batch phải hoàn thành trước downtime)
  * RC overflow prevention (RC stock tại thời điểm batch hoàn thành ≤ max_buffer)
* **R3 RC output direction:** $y_b \in \{0,1\}$ — mỗi batch PSC của R3 đổ RC vào Line 1 ($y_b = 1$) hay Line 2 ($y_b = 0$). Chỉ applicable cho batches assigned to R3. Trong experimental setting "fixed R3", $y_b = 0$ cho tất cả R3 batches (fixed to Line 2).

---

## **9. Reactive Scheduling Framework**

### **9.1 Ba Chiến Lược So Sánh**

| Chiến lược | Loại | Mô tả | Decision timing | Feasibility guarantee |
|-----------|------|-------|----------------|----------------------|
| **Dispatching Heuristic** | Rule-based (baseline) | Priority rules: EDD cho MTO, rồi fill-PSC cho stock. Đại diện cho cách operator hiện tại quyết định. | Instant | By construction (rules only select feasible actions) |
| **CP-SAT Re-solve** | Optimization-based | Khi UPS xảy ra: freeze batch đã hoàn thành, re-solve remaining horizon bằng CP-SAT. Tối ưu cho single re-plan instance. | ~0.1–1 second per re-solve | Guaranteed (solver enforces all constraints) |
| **DRL Policy (PPO)** | Learning-based | Agent train trên simulated UPS scenarios (MaskablePPO). Action masking đảm bảo feasibility. Response tức thì. Có thể học distributional patterns. | ~1 ms per inference | Via action masking (only feasible actions selectable) |

**Paradigm contrast — tại sao so sánh CP-SAT vs. DRL có ý nghĩa:**

CP-SAT re-solve là **stateless re-optimization**: mỗi lần UPS xảy ra, solver nhận current state, solve remaining horizon **from scratch**, không nhớ gì về UPS trước đó hay pattern nào. Mỗi re-solve là fresh optimization problem.

DRL policy là **stateful learned policy**: agent đã train trên hàng nghìn simulated shifts với UPS, và **đã học** implicit patterns — ví dụ "nếu Line 2 đã bị 2 UPS trong nửa đầu ca, nên route R3 output về Line 2 nhiều hơn vì Line 2 stock sẽ thấp hơn expected." CP-SAT không có khả năng này vì mỗi re-solve là independent.

Câu hỏi nghiên cứu: dưới disruption intensity nào, khả năng "nhớ" và "học pattern" của DRL vượt trội hơn khả năng "tối ưu exact" của CP-SAT?

### **9.2 Simulation Loop**

Cả ba chiến lược chạy trong cùng **simulation engine** — đảm bảo công bằng trong so sánh. Mỗi simulation run:

```
Input:
  - Shift parameters (roaster eligibility, PSC rate, planned downtime, MTO demand)
  - UPS realization (pre-generated list of (time, roaster, duration) triples)
  - Strategy choice (Dispatching / CP-SAT / DRL)
  - R3 routing mode (Fixed / Flexible)

Initialization:
  - Generate initial schedule:
      [CP-SAT]: Solve full 480-slot deterministic model
      [DRL]: Agent selects first action based on initial state
      [Dispatching]: Apply priority rules for first batch assignments
  - Set t = 0
  - Initialize roaster states: all IDLE (or RUNNING if initial batches started at t=0)
  - Initialize RC stock: B_l = B^0_l for each line

Main Loop — for t = 0 to 479:
  Step 1: Check UPS events
    - If UPS event at time t on roaster r:
        a. If r is RUNNING batch b:
           → Cancel batch b entirely
           → RC stock does NOT increase (batch not completed)
           → Pipeline time already consumed is sunk cost
        b. If r is SETUP:
           → Cancel setup, timer resets
        c. Set r → DOWN(remaining = d minutes)
        d. Trigger re-planning:
           [CP-SAT]: Freeze completed batches, re-solve [t, 479]
           [DRL]: Agent observes new state, selects action at next decision point
           [Dispatching]: No re-plan needed (rules are reactive by nature)
  
  Step 2: Advance roaster states
    - For each roaster r:
        If RUNNING: decrement remaining_time. If remaining = 0:
           → Batch complete. RC stock += 1 (if PSC and out(b) = l).
           → Roaster → IDLE. This is a DECISION POINT.
        If SETUP: decrement remaining_time. If remaining = 0:
           → Roaster → IDLE. This is a DECISION POINT.
        If DOWN: decrement remaining_time. If remaining = 0:
           → Roaster → IDLE. This is a DECISION POINT.
  
  Step 3: Apply consumption events
    - If t ∈ E_l for line l:
        B_l -= 1
        If B_l < 0: record stockout event
  
  Step 4: Decision points
    - For each roaster r that just became IDLE:
        [CP-SAT]: Execute next batch from current schedule
        [DRL]: Agent observes state, selects (SKU, roaster) or WAIT
        [Dispatching]: Apply rules:
           1. If MTO remaining AND time allows: schedule MTO batch
           2. Else: schedule PSC batch
           3. If pipeline busy: WAIT until free

  Step 5: t += 1

Output:
  - Total PSC batches completed (per line and total)
  - Stockout events (count and duration per line)
  - MTO tardiness (per job)
  - Re-solve count (CP-SAT only)
  - Computation time per decision
```

### **9.3 State Vector**

Tại mỗi decision point hoặc re-planning trigger $t_0$, trạng thái hệ thống:

| Component | Type | Mô tả | Range |
|-----------|------|-------|-------|
| $t_0$ | int | Time slot hiện tại | [0, 479] |
| $\text{status}_r$ | enum | Trạng thái roaster $r$ | {IDLE, RUNNING, DOWN, SETUP} |
| $\text{remaining}_r$ | int | Phút còn lại trong trạng thái hiện tại (0 nếu IDLE) | [0, ~30] |
| $\text{current\_sku}_r$ | SKU | SKU cuối cùng được xử lý trên roaster $r$ | {PSC, NDG, Busta, None} |
| $B_l(t_0)$ | int | RC stock hiện tại trên line $l$ | $[0, \overline{B}_l]$ |
| $\text{MTO\_remaining}_j$ | int | Số batch MTO còn lại cho job $j$ | [0, initial_count] |
| $\text{pipeline\_busy}_l$ | int | Phút còn lại pipeline bận trên line $l$ (0 nếu free) | [0, 2] |

Cho **CP-SAT re-solve**: state → initial conditions của reduced model trên $[t_0, 479]$.
Cho **DRL**: state → observation vector (normalized to [0,1]) fed vào policy network.
Cho **Dispatching**: state → inputs cho priority rules.

### **9.4 Decision Points — Event-Driven, Không Phải Mỗi Time Slot**

DRL agent và dispatching heuristic **không** quyết định mỗi time slot (480 decisions quá nhiều). Thay vào đó, quyết định chỉ xảy ra tại **event-driven decision points**:

1. Roaster trở thành IDLE (batch hoàn thành, setup hoàn thành, hoặc UPS kết thúc)
2. Pipeline trở thành free (consume operation hoàn thành trên line có roaster đang chờ)

Giữa các decision points, simulation tự động advance. Điều này giảm effective episode length từ 480 xuống khoảng **30–60 decisions per shift**, phù hợp hơn cho RL training.

---

## **10. Experimental Design**

### **10.1 Factorial Design**

| Factor | Levels | Count | Rationale |
|--------|--------|-------|-----------|
| UPS rate $\lambda$ | 0, 1, 2, 3, 5 events/shift | 5 | Từ không có disruption đến disruption nặng |
| UPS duration $\mu$ | 10, 20, 30 min (mean) | 3 | Từ sửa nhanh đến sửa lớn |
| Strategy | Dispatching / CP-SAT / DRL | 3 | Ba paradigm khác nhau |
| R3 routing | Fixed (Line 2 only) / Flexible | 2 | Đo giá trị cross-line flexibility |

Full factorial: 5 × 3 × 3 × 2 = **90 cells**.
Replications per cell: **100** (khác nhau về UPS realization — cùng $\lambda$, $\mu$ nhưng random seed khác → timing và roaster bị ảnh hưởng khác nhau).
**Total: 9,000 simulation runs.**

### **10.2 Controlled Randomness**

Mỗi cell 100 replications sử dụng **cùng 100 UPS realizations** cho cả 3 strategies. Nghĩa là: replication #37 trong cell ($\lambda=3$, $\mu=20$) có **chính xác cùng UPS events** (cùng timing, cùng roaster, cùng duration) cho Dispatching, CP-SAT, và DRL. Sự khác biệt trong kết quả **chỉ** đến từ strategy, không phải randomness. Đây là **paired comparison** — giảm variance và tăng statistical power.

### **10.3 KPIs**

| KPI | Định nghĩa | Đơn vị | Mục đích |
|-----|-----------|--------|----------|
| **Total throughput** | Tổng PSC batch hoàn thành (cả 2 lines) | batches | Primary performance metric |
| **Stockout count** | Số consumption events với $B_l \leq 0$ | events | Severity of inventory failure |
| **Stockout duration** | Tổng thời gian $B_l \leq 0$ trên tất cả lines | minutes | Duration of service disruption |
| **MTO tardiness** | $\sum_j \text{tard}_j$ | minutes | Due date compliance |
| **Re-solve count** | Số lần CP-SAT được gọi (chỉ CP-SAT strategy) | count | Computational overhead |
| **Computation time** | Wall-clock time per re-solve (CP-SAT) / per inference (DRL) | seconds | Real-time deployability |

---

## **11. Assumptions**

| ID | Assumption | Chi tiết | Risk nếu violated |
|----|-----------|---------|-------------------|
| A1 | Roasting time đồng nhất 15 phút | Thực tế dao động 13–16 phút tùy SKU | Pipeline utilization thay đổi; Line 2 có thể tighter hoặc looser |
| A2 | Setup time đồng nhất 5 phút | Mọi cặp SKU transition tốn 5 phút | Một số transition thực tế nhanh/chậm hơn → sequencing priority thay đổi |
| A3 | GC supply unlimited | Không theo dõi GC silo inventory | Chấp nhận khi thesis focus là scheduling methodology; GC shortage hiếm trong 1 ca nếu được chuẩn bị trước |
| A4 | PSC consumption rate cố định | Rate không đổi suốt ca | Biến động rate (line speed adjustment) không capture |
| A5 | UPS parameters synthetic | Literature-based MTBF/MTTR, không calibrate từ plant data | Kết quả cho relative performance, không absolute prediction |
| A6 | UPS ảnh hưởng đúng 1 roaster | Không model correlated failures | Mất điện, compressed air failure ảnh hưởng nhiều roasters → underestimate disruption |
| A7 | UPS hủy batch hoàn toàn | Không pause/resume. GC mất, restart từ đầu | Slightly pessimistic nếu thực tế cho phép resume partial roast |
| A8 | Không xét shelf life và FIFO | Chấp nhận trong 8 giờ | Nếu multi-shift, cần extend |
| A9 | RC là aggregate counter | Không theo dõi từng RC silo riêng | Chấp nhận khi chỉ 1 PSC SKU per shift — không mixing concern |

---

## **12. Out of Scope**

* GC silo inventory management (capacity, SKU purity, dump, replenish, dynamic SKU reassignment)
* RC individual silo-level tracking (fill rules, draw rules, buffer silo logic)
* PSC-PSC SKU switching within a shift
* 4-hour changeover (full line changeover with extended downtime)
* Bill of Materials tracking (GC quantities per batch)
* Shelf life và FIFO theo date
* Tối ưu hóa chất lượng rang (roast profile optimization)
* Giới hạn nhân công (labor constraints)
* Stochastic demand (PSC consumption rate là deterministic input)
* Multi-shift planning horizon (cross-shift optimization)
* Cross-shift inventory carryover (mỗi shift là independent instance)
* Sequence-dependent setup time matrix (dùng giá trị đồng nhất 5 phút thay vì matrix per SKU pair)
* Proactive robust scheduling (schedule designed to be robust against future UPS without knowing when they occur)
* Rolling horizon decomposition (solving in windows rather than full horizon)
* Metaheuristic approaches (ALNS, GA — documented as potential future work)
