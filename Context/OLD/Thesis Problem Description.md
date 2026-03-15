# **Problem Statement — Updated Complete Version**

---

## **1\. Tổng Quan Bài Toán**

Bài toán đặt ra là xây dựng một hệ thống lập lịch tự động cho quá trình rang cà phê theo batch tại nhà máy Nestlé Trị An, trong phạm vi một ca làm việc kéo dài 8 giờ, được discretize thành **480 time slot, mỗi slot \= 1 phút**. Đây là bài toán lập lịch đa máy, đa sản phẩm, trong đó có hai lớp ràng buộc cốt lõi đan xen nhau.

Lớp thứ nhất là **hệ thống GC silo hữu hạn với đường ống xả dùng chung per line** — tạo ra bottleneck vật lý buộc các roaster trên cùng line phải cạnh tranh quyền truy cập nguyên liệu tại mỗi time slot. Lớp thứ hai là **sequence-dependent setup time trên roaster** — mỗi khi roaster chuyển sang rang SKU khác với batch vừa hoàn thành, cần 5 phút setup bắt buộc, tạo ra áp lực tự nhiên để scheduler nhóm các batch cùng SKU lại với nhau thay vì xen kẽ.

Hệ thống phải đồng thời giải quyết ba bài toán con lồng vào nhau. Thứ nhất, đáp ứng các đơn hàng NDG và Busta theo kiểu make-to-order với thời hạn giao hàng trong 4 giờ đầu ca. Thứ hai, duy trì tồn kho RC cho PSC ở mức ổn định để đáp ứng lịch tiêu thụ deterministic của dây chuyền đóng gói, tránh tình trạng thiếu hàng gây dừng line. Thứ ba, quản lý toàn bộ vòng đời của GC silo — từ việc consume nguyên liệu cho từng batch, đến quyết định khi nào replenish, khi nào dump, và silo nào nên được chuẩn bị sẵn cho SKU switch hoặc changeover trong tương lai — trong điều kiện đường ống xả của mỗi line chỉ phục vụ được một thao tác tại một thời điểm.

Mục tiêu tối ưu là **tối thiểu hóa tổng chi phí penalty quy đổi ra VND** trong toàn ca, bao gồm chi phí thiếu hàng PSC, chi phí roaster idle không hiệu quả, chi phí dump nguyên liệu, chi phí replenish, và chi phí trễ hạn giao hàng NDG/Busta.

---

## **2\. Cấu Hình Hệ Thống Vật Lý**

### **2.1 Hai Line Sản Xuất**

Nhà máy có hai line sản xuất hoạt động song song. Hai line độc lập hoàn toàn về GC silo và RC silo, ngoại trừ Roaster 3 là điểm kết nối chéo duy nhất giữa hai line.

**Line 1** bao gồm Roaster 1 và Roaster 2, với hệ thống 8 GC silo riêng và 4 RC silo riêng.

**Line 2** bao gồm Roaster 3, Roaster 4 và Roaster 5, với hệ thống 8 GC silo riêng và 4 RC silo riêng.

### **2.2 Năm Roasters — Compatibility, Sequencing và Setup Time**

Mỗi roaster xử lý tối đa 1 batch tại một thời điểm. Thời gian rang của mỗi batch là cố định theo SKU, dao động từ 13 đến 16 phút, và được biểu diễn là số nguyên time slot.

* **Roaster 1:** Rang được NDG và PSC. Consume GC từ Line 1 silos. RC output đổ vào RC silo của Line 1\.  
* **Roaster 2:** Rang được NDG, Busta và PSC. Consume GC từ Line 1 silos. RC output đổ vào RC silo của Line 1\.  
* **Roaster 3:** Chỉ rang PSC. Luôn consume GC từ Line 2 silos. RC output có thể đổ vào RC silo của Line 1 hoặc Line 2, tùy quyết định của scheduler cho từng batch, không tốn thêm thời gian chuyển đổi.  
* **Roaster 4:** Chỉ rang PSC. Consume GC từ Line 2 silos. RC output đổ vào RC silo của Line 2\.  
* **Roaster 5:** Chỉ rang PSC. Consume GC từ Line 2 silos. RC output đổ vào RC silo của Line 2\.

Roaster 3 là roaster duy nhất tạo ra luồng vật chất chéo: lấy nguyên liệu từ Line 2 nhưng có thể bổ sung tồn kho cho Line 1\. Đây là điểm linh hoạt quan trọng trong vận hành nhưng cũng là nguồn phức tạp trong lập lịch.

**Sequence-Dependent Setup Time — Ràng Buộc Cứng:**

Khi hai batch liên tiếp trên cùng một roaster có SKU khác nhau, bắt buộc phải có **5 time slot (5 phút) setup time** giữa thời điểm kết thúc batch trước và thời điểm bắt đầu consume của batch sau. Trong 5 phút này, roaster không làm gì cả — không rang, không consume GC. Đường ống GC của line vẫn tự do và có thể phục vụ replenish, dump, hoặc consume cho roaster khác trên cùng line trong khoảng thời gian này.

Setup time áp dụng cho **mọi loại transition SKU** mà không phân biệt: NDG-A sang NDG-B, NDG sang PSC, PSC-R1 sang PSC-R2, hay bất kỳ cặp SKU nào khác — chỉ cần SKU(batch i) ≠ SKU(batch i+1) trên cùng roaster thì bắt buộc chờ 5 phút. Ngược lại, nếu hai batch liên tiếp cùng SKU, không có setup time — batch sau có thể bắt đầu consume ngay khi đường ống rảnh sau khi batch trước kết thúc.

Hệ quả thực tế của ràng buộc này là scheduler bị khuyến khích một cách tự nhiên để **nhóm tất cả batch cùng SKU lại liên tiếp** trên cùng một roaster. Ví dụ: nếu ticket NDG-A cần 3 batch trên R1, việc sắp xếp 3 batch này liên tiếp nhau sẽ không tốn setup time. Nếu xen vào 1 batch NDG-B ở giữa, scheduler sẽ phải trả 2 lần 5 phút setup — 10 phút lãng phí năng lực roaster. Không có penalty riêng cho việc không nhóm batch: chi phí đã được capture hoàn toàn bởi setup time làm tăng total completion time và có thể gây trễ due date hoặc tăng idle penalty.

Ví dụ minh họa cụ thể: R1 hoàn thành batch NDG-A cuối cùng tại phút 45\. Batch tiếp theo trên R1 là PSC-R1 — vì SKU khác, R1 phải chờ đến phút 50 mới được consume GC cho batch PSC-R1. Nếu batch tiếp theo vẫn là NDG-A, R1 có thể consume ngay từ phút 45 (hoặc ngay khi đường ống rảnh, tùy trạng thái pipeline).

### **2.3 Hệ Thống GC Silo**

Mỗi line có **8 GC silo**, mỗi silo capacity tối đa **3,000 kg**. GC silo lưu trữ green coffee beans — nguyên liệu thô được tipping từ kho lớn xuống silo, và từ silo được consume trực tiếp vào roaster khi bắt đầu mỗi batch.

**Ràng buộc cứng về nội dung silo:**

* Single-SKU per silo: tại mọi thời điểm, một silo chỉ chứa tối đa một loại GC-SKU.  
* No mixing: không được replenish GC-SKU khác vào silo đang chứa SKU hiện tại. Muốn đổi SKU trong 1 silo, silo phải được empty về 0 kg trước.  
* No overfill: level không vượt 3,000 kg.  
* No underflow: level không âm. Scheduler không được assign batch lấy GC từ silo nếu silo không đủ lượng theo BOM.

**SKU assignment của silo là động trong ca:** Khi silo về 0 kg, scheduler có thể assign SKU mới thông qua quyết định replenish tiếp theo. Đây là quyết định của model, không cố định.

**Đường ống xả dùng chung — ràng buộc vật lý quan trọng nhất của GC silo:**

Mỗi line có một đường ống xả duy nhất kết nối toàn bộ 8 silo của line đó. Ba loại thao tác có thể xảy ra trên đường ống này là replenish, consume và dump. Ba thao tác này **mutually exclusive** tại mỗi time slot — tại bất kỳ thời điểm nào, đường ống của line chỉ có thể đang thực hiện đúng một trong ba loại thao tác. Nếu đường ống đang bận, không thao tác nào khác có thể bắt đầu cho đến khi thao tác hiện tại hoàn thành. Thời gian của mỗi thao tác như sau:

* **Replenish:** kéo dài **2 time slot** (2 phút). Trong 2 slot này, không thể consume hay dump bất kỳ silo nào của line đó. Đây là thao tác nạp GC từ kho lớn xuống 1 silo, với lot size cố định 500 kg mỗi lần.

* **Consume:** kéo dài **2 time slot** (2 phút, tương đương 100 giây thực tế). Đây là thao tác lấy toàn bộ GC cần thiết theo BOM cho 1 batch — toàn bộ GC-SKU cần cho batch đó được lấy trong cùng 1 lần consume duy nhất, không lấy từng GC-SKU riêng biệt qua nhiều lần. Hệ thống cấp GC đang phục vụ 1 batch và có thể lấy đủ các thành phần BOM trong một chu kỳ 2 phút (coi như thao tác gộp).Thời điểm batch bắt đầu rang đồng thời là thời điểm consume bắt đầu. Nếu đường ống đang bận khi roaster đã sẵn sàng consume (ví dụ đang replenish hoặc đang dump), roaster phải đợi đến khi đường ống rảnh — đây là nguồn phát sinh delay và idle time nằm ngoài tầm kiểm soát trực tiếp của roaster. 

* **Dump:** kéo dài **5 time slot cố định cộng thêm 1 time slot cho mỗi 100 kg bị dump**. Ví dụ: dump 550 kg tốn 5 \+ ceil(550/100) \= 5 \+ 6 \= 11 time slot. Trong toàn bộ thời gian dump, không thể replenish hay consume bất kỳ thao tác nào khác trên đường ống của line đó.

**Hệ quả vận hành quan trọng:** Trên Line 1, nếu R1 đang trong setup time (chờ 5 phút sau khi đổi SKU) và R2 cần consume ngay lúc đó, R2 có thể consume bình thường vì setup time của R1 không chiếm đường ống. Tuy nhiên, nếu R2 đang consume (2 phút) thì R1 dù đã hết setup time vẫn phải đợi thêm cho đến khi R2 consume xong. Hai nguồn delay này — setup time và pipeline contention — hoạt động độc lập nhau và có thể cộng dồn.

**Điều kiện consume khả thi:** Trước khi thực hiện consume cho 1 batch, tổng GC trong các silo được chỉ định phải đủ toàn bộ lượng yêu cầu theo BOM. Nếu không đủ (do chưa được replenish kịp), roaster phải idle tại time slot đó cho đến khi đủ điều kiện.

### **2.4 Hệ Thống RC Silo**

Mỗi line có **4 RC silo**, mỗi silo capacity tối đa **5,000 kg**, tổng capacity per line là **20,000 kg** (hard). RC silo lưu trữ roasted coffee chờ cấp cho dây chuyền PSC đóng gói.

**Ràng buộc cứng về nội dung RC silo:**

* Single-SKU per RC silo, no mixing — tương tự GC silo.  
* Mỗi RC silo không được vượt 5,000 kg.  
* Tổng tất cả RC silo của line không vượt 20,000 kg.  
* RC level không âm.

**Quy tắc fill RC silo khi batch hoàn thành rang:**

Output của batch PSC được đổ vào RC silo theo quy tắc ưu tiên silo đang chứa đúng SKU đó và có stock thấp nhất (lowest stock first). Trong thực tế vận hành, 3 trong 4 RC silo được fill trước theo lowest stock first — silo thứ 4 được dùng như buffer dự phòng và chỉ được fill khi cả 3 silo kia đều đầy. Nếu tất cả RC silo đang chứa đúng SKU đó đều đầy (5,000 kg), batch không được phép start vì RC output không có chỗ chứa — đây là hard constraint look-ahead feasibility. 

**Quy tắc PSC consumption từ RC silo theo FIFO:**

PSC consumption ưu tiên lấy từ silo thứ 4 (buffer silo) trước. Nếu silo thứ 4 empty hoặc không chứa đúng current RC-SKU đang chạy (ví dụ sau SKU switch), PSC lấy từ 3 silo còn lại theo thứ tự silo được fill sớm nhất trước (FIFO). PSC lấy hết hoàn toàn 1 silo rồi mới chuyển sang silo kế tiếp — không lấy song song từ nhiều silo cùng lúc.

**Quy tắc chuyển SKU trong RC consumption:**

PSC chỉ bắt đầu consume SKU mới sau khi toàn bộ SKU cũ đã được tiêu thụ hết trong tất cả RC silo. Không có mixing giữa 2 SKU trong quá trình consumption — đây là hard constraint.

**RC silo không có mutual exclusion:** Roaster có thể đổ RC output vào RC silo tại cùng thời điểm PSC đang consume RC từ silo khác. Không có đường ống dùng chung cho RC.

---

## **3\. Sản Phẩm, Jobs và Batches**

### **3.1 NDG và Busta — Make-to-Order**

NDG và Busta là sản phẩm sản xuất theo đơn hàng. Toàn bộ thông tin ticket đã biết từ đầu ca. Mỗi ticket gồm loại RC-SKU, số batch cần rang, batch size cố định theo SKU (kg/batch), và roasting time cố định theo SKU (phút/batch).

Mỗi ticket là 1 job gồm nhiều batch. Do setup time 5 phút được áp dụng mỗi khi roaster đổi SKU, scheduler bị khuyến khích tự nhiên để chạy toàn bộ batch của cùng 1 job liên tiếp nhau trên cùng 1 roaster. Đây không còn là soft constraint với penalty riêng — chi phí của việc phân tán batch đã được phản ánh hoàn toàn thông qua setup time làm mất thêm thời gian rang và có thể gây trễ due date.

**Due date:** Toàn bộ batch của mỗi ticket NDG/Busta phải hoàn thành trong **4 giờ đầu ca**. Đây là soft constraint với penalty rất lớn, calibrate để hành xử như hard constraint trong hầu hết tình huống. Lý do không dùng hard constraint thuần túy là để tránh model báo infeasible trong các tình huống cực đoan — thay vào đó model báo cáo mức độ vi phạm để operator xử lý.

**RC output:** NDG và Busta không đổ vào RC silo — giao thẳng, không qua inventory.

### **3.2 PSC — Make-to-Stock**

PSC là sản phẩm sản xuất để bổ sung tồn kho RC. Không có ticket cố định — scheduler tự quyết định batch PSC nào cần chạy, khi nào, trên roaster nào, để duy trì RC đủ đáp ứng consumption plan trong suốt ca.

### **3.3 BOM — Bill of Materials**

Mỗi RC-SKU cần 1 đến 3 GC-SKU với tỷ lệ cố định (BOM ratio), là input cố định của bài toán. Ví dụ: để rang 500 kg RC-SKU-A cần 300 kg GC-SKU-X và 200 kg GC-SKU-Y (tỷ lệ 60/40).

**Quy tắc sourcing GC cho từng batch:**

Mặc định, mỗi GC-SKU trong BOM của 1 batch được lấy từ đúng 1 silo duy nhất. Ngoại lệ duy nhất được phép là lấy từ 2 silo, nhưng chỉ khi và chỉ khi ít nhất 1 trong 2 silo đó có thể được emptied hoàn toàn bởi thao tác lấy đó. Mục đích duy nhất của việc split là giải phóng silo để chuẩn bị cho SKU mới hoặc giảm số silo bị chiếm dụng. Trong mọi trường hợp, tổng GC lấy ra phải đúng bằng lượng yêu cầu theo BOM. Toàn bộ việc lấy GC cho 1 batch — dù từ 1 hay 2 silo, dù có 1 hay 3 GC-SKU trong BOM — diễn ra trong **1 consume event duy nhất kéo dài 2 time slot** trên đường ống.

---

## **4\. Hai Loại Đổi SKU PSC**

Đây là điểm phức tạp và quan trọng nhất trong vận hành PSC, đòi hỏi phân biệt rõ ràng hai loại sự kiện có tên giống nhau nhưng bản chất hoàn toàn khác nhau.

### **4.1 SKU Switch — Đổi SKU Thông Thường**

SKU switch là sự kiện thay đổi RC-SKU đang sản xuất trong điều kiện hai SKU có thuộc tính cà phê tương đồng — cùng roast profile hoặc chỉ khác biệt nhỏ về blend. SKU switch có thể xảy ra thường xuyên, có thể nhiều lần trong 1 ca hoặc giữa 2 ca liên tiếp, và hoàn toàn theo plan.

**Điều kiện cứng trước SKU switch:** Tổng RC của SKU-A trong tất cả RC silo của line đó phải **dưới 15,000 kg** tại thời điểm batch đầu tiên của RC-SKU-B bắt đầu đổ output vào RC silo — đảm bảo còn ít nhất 1 RC silo trống (5,000 kg) để nhận SKU-B mà không mixing với SKU-A còn lại. Không có downtime của roaster khi switch.

**Lưu ý về setup time khi switch:** Tại thời điểm switch, roaster nào đang rang SKU-A mà chuyển sang rang SKU-B sẽ phải chịu 5 phút setup time như bình thường. Đây không phải chi phí đặc biệt của switch — chỉ là sequence-dependent setup time áp dụng nhất quán cho mọi SKU change trên bất kỳ roaster nào.

**Chuẩn bị GC silo trước switch:** Nếu SKU-B cần GC-SKU-Z mà hiện tại không có silo nào chứa GC-SKU-Z, model phải chủ động dump silo phù hợp và replenish GC-SKU-Z trước thời điểm switch. Model biết trước thời điểm switch từ production plan và phải tính toán lead time của dump \+ replenish để đảm bảo GC đủ sẵn sàng.

### **4.2 Changeover — Đổi SKU Có Downtime**

Changeover là sự kiện thay đổi RC-SKU khi hai SKU có thuộc tính căn bản khác nhau — ví dụ chuyển từ Light Roast sang Dark Roast, hoặc từ cà phê thường sang Soya blend. Changeover xảy ra rất ít (cách nhau 4 đến 10 ngày), hoàn toàn theo plan, và model nhận đây như input cố định.

**Điều kiện cứng trước changeover:** RC của SKU cũ phải dưới 15,000 kg tại thời điểm bắt đầu changeover.

**Thời điểm và duration:** Changeover luôn xảy ra đúng tại ranh giới nửa ca (giờ 0 hoặc giờ 4), kéo dài **4 giờ** (1 nửa ca hoàn chỉnh). Mỗi line tối đa 1 changeover per ca. Changeover của 2 line độc lập nhau.

**Quy trình trong 4 giờ changeover của 1 line:**

Ngay khi changeover bắt đầu, không được start thêm batch PSC nào có RC output vào line đang changeover. Các batch đang rang dở hoàn thành nốt rồi dừng. RC của SKU cũ tiếp tục được PSC consumption tiêu thụ dần về 0 — không tính shortage penalty và không tính idle penalty trong giai đoạn này vì đây là trạng thái bắt buộc theo plan.

Với Roaster 3, khi changeover Line 1: R3 hoàn thành batch hiện tại (nếu output đang vào Line 1\) rồi tự động chuyển output sang Line 2, và có thể tiếp tục rang cho Line 2 nếu GC Line 2 còn đủ. Khi changeover Line 2: R3 hoàn thành batch hiện tại cho Line 2 rồi có thể switch output sang Line 1 nếu cần.

GC silo của line đang changeover phải được model xử lý chủ động: biết trước thời điểm changeover và BOM của SKU mới, model tính toán cách dump và replenish sao cho khi changeover kết thúc, lượng GC không cần thiết còn lại trong silo là ít nhất. Chi phí dump và replenish vẫn tính bình thường.

Sau khi changeover hoàn tất, line bắt đầu rang PSC SKU mới. PSC consumption của SKU mới bắt đầu sau **thêm 2 giờ** kể từ khi changeover kết thúc (thời gian dây chuyền PSC chuyển đổi).

Ca có changeover sẽ không có ticket NDG/Busta theo lịch sản xuất — đây là input condition, không phải model constraint.

---

## **5\. PSC Consumption**

PSC tiêu thụ RC theo lịch deterministic. Mỗi RC-SKU được gán tốc độ tiêu thụ cố định là **X kg mỗi 10 phút**, trong đó 10 phút là cycle time cố định của dây chuyền PSC cho mọi loại cà phê, còn X tùy theo từng RC-SKU và là input của bài toán.

Consumption là discrete event: tại mỗi mốc 10 phút, nếu RC-SKU đó đang là current running SKU của line, hệ thống trừ X kg khỏi RC silo tương ứng theo quy tắc FIFO đã mô tả ở mục 2.4. RC không được âm — đây là hard constraint. Nếu tại mốc consumption mà RC không đủ X kg, phần thiếu phát sinh PSC shortage penalty.

PSC chỉ bắt đầu consume SKU mới sau SKU switch hoặc changeover khi toàn bộ SKU cũ đã hết trong tất cả RC silo.

---

## **6\. Planned Downtime**

Planned downtime là các khoảng thời gian đã lên lịch trước mà roaster không thể hoạt động. Thông tin này là input cố định, biết từ đầu ca. Downtime có thể khai báo per roaster, per line (R1–R2 hoặc R3–R5), hoặc toàn nhà máy (R1–R5).

Trong khoảng downtime, roaster không được bắt đầu batch mới và không được trong trạng thái setup time. Batch đang rang khi downtime bắt đầu được hoàn thành nốt — downtime không cắt ngang batch đang chạy. Unplanned breakdown hoàn toàn ngoài phạm vi model.

---

## **7\. Objective Function — Tối Thiểu Hóa Tổng Penalty (VND)**

Model tối thiểu hóa tổng penalty cost trong toàn ca 8 giờ, gồm các thành phần sau.

**PSC Shortage Penalty:** Phát sinh khi RC không đủ tại mốc consumption event. Chi phí **600,000 VND cho mỗi phút thiếu hàng**. Không áp dụng trong giai đoạn changeover.

**Idle Penalty:** Phát sinh khi roaster idle trong khi RC của line đó đang dưới ngưỡng safety stock **10,000 kg**. Chi phí **100,000 VND cho mỗi phút roaster idle khi RC(line) \< 10,000 kg**. Idle do setup time (5 phút sau đổi SKU) vẫn tính vào idle penalty nếu RC dưới safety stock — đây là thêm một lý do để scheduler tránh đổi SKU không cần thiết khi tồn kho thấp. Không áp dụng trong giai đoạn changeover.

**Dump Penalty:** Phát sinh mỗi khi thực hiện dump GC. Chi phí **500,000 VND cố định mỗi lần dump cộng thêm 10,000 VND cho mỗi kg bị dump**.

**Replenish Penalty:** Phát sinh mỗi khi thực hiện replenish vào GC silo. Chi phí **100,000 VND cố định mỗi lần replenish**.

**NDG/Busta Tardiness Penalty:** Phát sinh khi batch cuối cùng của ticket NDG/Busta hoàn thành sau mốc 4 giờ. Chi phí là một penalty rất lớn nhân với số phút trễ, calibrate đủ lớn để model ưu tiên đảm bảo due date trong hầu hết tình huống.

---

## **8\. Decision Variables**

Model tự quyết định toàn bộ các biến sau.

* **Roaster assignment và start time:** Mỗi batch chạy trên roaster nào, bắt đầu tại time slot nào. Start time phải tôn trọng setup time 5 phút nếu batch trước trên cùng roaster có SKU khác.  
* **R3 RC output direction:** Mỗi batch của R3 đổ RC vào Line 1 hay Line 2\.  
* **PSC batch selection:** Cần rang bao nhiêu batch PSC của từng SKU, trên roaster nào, tại time slot nào.  
* **GC silo sourcing:** Với mỗi GC-SKU trong BOM của mỗi batch, lấy từ silo nào (1 silo default, hoặc 2 silo nếu có thể empty ít nhất 1).  
* **Replenish decisions:** Khi nào replenish, GC-SKU nào, vào silo nào.  
* **Dump decisions:** Silo nào, bao nhiêu kg, khi nào.  
* **Silo SKU reassignment:** Khi silo GC về 0, assign SKU mới nào thông qua replenish tiếp theo.  
* **RC silo fill assignment:** Batch PSC hoàn thành thì đổ vào RC silo nào.

---

## **9\. Assumptions và Limitations**

**A1 — Determinism hoàn toàn:** Tất cả input đều deterministic và biết từ đầu ca. Nếu có unplanned breakdown hoặc plan thay đổi giữa ca, cần re-plan thủ công.

**A2 — Thời gian thao tác GC silo được discretize:** Replenish 2 slot, consume 2 slot, dump theo công thức. Trong thực tế có thể có sai lệch nhỏ so với thời gian vật lý.

**A3 — PSC consumption discrete mỗi 10 phút:** Cycle time PSC cố định 10 phút cho mọi SKU. Không phù hợp nếu cycle time thay đổi trong thực tế.

**A4 — Không xét shelf life và FIFO theo date:** Chấp nhận được trong phạm vi 1 ca 8 giờ.

**A5 — NDG/Busta soft due date:** Penalty calibrate đủ lớn để effective-hard trong hầu hết tình huống.

**A6 — Dump instant về mặt lượng:** Toàn bộ lượng dump được tính giảm ngay tại time slot đầu tiên của dump event, mặc dù duration của dump kéo dài nhiều slot. Điều này đơn giản hóa inventory balance.

**A7 — Setup time đồng nhất cho mọi transition:** 5 phút áp dụng cho tất cả cặp SKU bất kể mức độ tương đồng. Trong thực tế, một số transition có thể cần ít hoặc nhiều hơn 5 phút. Nếu có dữ liệu thực tế về setup time matrix (thời gian khác nhau cho từng cặp SKU), model có thể mở rộng để sử dụng sequence-dependent setup time matrix thay vì giá trị đồng nhất.

---

## **10\. Out of Scope**

Unplanned breakdown, shelf life và FIFO theo date, tối ưu hóa chất lượng rang, giới hạn nhân công, stochastic demand, real-time re-scheduling giữa ca, sequence-dependent setup time matrix (hiện tại dùng giá trị đồng nhất 5 phút).

