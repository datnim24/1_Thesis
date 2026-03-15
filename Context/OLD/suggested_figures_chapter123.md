# Suggested Figures, Tables & Diagrams — Chapters 1, 2, 3
## Thesis: Automated Batch Roasting Scheduling at Nestlé Trị An

> **Legend:**
> - 🔴 **Bắt buộc** — committee sẽ hỏi nếu thiếu
> - 🟡 **Nên có** — làm mạnh hơn nhưng không block việc viết
> - 🟢 **Nice to have** — chỉ làm nếu còn thời gian
> - **Loại:** Figure (F) / Table (T)
> - **Nguồn:** Tự vẽ / Từ data / Từ solver output / Public

---

## Chapter 1 — Introduction

### Section 1.1 — Background

| ID | Tiêu đề | Mô tả nội dung | Loại | Ưu tiên | Công cụ gợi ý | Nguồn |
|---|---|---|---|---|---|---|
| F1.1 | **Sơ đồ vật lý tổng quan nhà máy** | 2 lines song song, 5 roasters (R1–R5), 16 GC silos, 8 RC silos, PSC line. Thể hiện R3 là điểm nối chéo duy nhất. Đây là hình đầu tiên committee nhìn vào để hiểu hệ thống | Diagram | 🔴 | draw.io | Tự vẽ |
| F1.2 | **Luồng vật liệu (Material Flow)** | GC Warehouse → GC Silo → Roaster → RC Silo → PSC Packaging Line. Dạng flowchart đơn giản, trái sang phải. Phân biệt luồng GC (xanh) và luồng RC (đỏ) | Flowchart | 🔴 | draw.io / PowerPoint | Tự vẽ |
| F1.3 | **R3 Cross-Line Routing Detail** | Highlight riêng R3: mũi tên GC từ L2 silos vào R3, sau đó RC output có 2 mũi tên (một sang L1, một sang L2) với label "scheduling decision per batch". Khác biệt hoàn toàn với R1/R2/R4/R5 | Diagram | 🟡 | draw.io | Tự vẽ |

---

### Section 1.2 — Problem Statement

| ID | Tiêu đề | Mô tả nội dung | Loại | Ưu tiên | Công cụ gợi ý | Nguồn |
|---|---|---|---|---|---|---|
| F1.4 | **Ba sub-problem và sự thiếu phối hợp** | 3 ô riêng biệt: (1) MTO Sequencing, (2) MTS Replenishment, (3) GC Silo Management. Giữa chúng: mũi tên đứt đoạn label "currently uncoordinated". Bên phải: mũi tên hội tụ vào 1 ô "Integrated CP Model (this thesis)" | Conceptual diagram | 🔴 | draw.io / PowerPoint | Tự vẽ |
| T1.1 | **Bảng tóm tắt pain points quan sát được** | 3 cột: Observation / Rough Figure / Implication. 3 dòng: unplanned dumps (~3/week, 2hr handling), RC high-stock stoppage (~300 min), speed loss (~18%). Chú thích rõ: "estimated from on-site observation, 1–2 samples" | Table | 🟡 | Word / LaTeX | Từ Q&A |

---

### Section 1.3 — Objectives

| ID | Tiêu đề | Mô tả nội dung | Loại | Ưu tiên | Công cụ gợi ý | Nguồn |
|---|---|---|---|---|---|---|
| F1.5 | **System Input–Process–Output Diagram** | Inputs (trái): BOM ratios, initial silo states, demand tickets (MTO), PSC consumption schedule, planned downtime. Process (giữa): Optimization Model (MILP / CP-SAT). Outputs (phải): shift schedule (roaster assignment + timing), GC operation plan, penalty score Z. Dạng hộp IPO chuẩn | I/O Diagram | 🔴 | draw.io / PowerPoint | Tự vẽ |

---

### Section 1.4 — Scope and Limitation

| ID | Tiêu đề | Mô tả nội dung | Loại | Ưu tiên | Công cụ gợi ý | Nguồn |
|---|---|---|---|---|---|---|
| F1.6 | **Scope Boundary Diagram** | Hộp lớn bên trong = "In Scope": 1 shift (480 min), all 3 shift types, offline pre-shift planning, MILP + CP-SAT. Bên ngoài hộp = "Out of Scope": real-time rescheduling, multi-shift, weekly planning, metaheuristics, deployment. Rõ ràng bằng màu sắc | Boundary diagram | 🟡 | draw.io / PowerPoint | Tự vẽ |

---

## Chapter 2 — Related Works

### Section 2.1 — Company Introduction

| ID | Tiêu đề | Mô tả nội dung | Loại | Ưu tiên | Công cụ gợi ý | Nguồn |
|---|---|---|---|---|---|---|
| F2.1 | **Ảnh / Logo Nestlé Trị An** | Ảnh nhà máy hoặc logo chính thức Nestlé Vietnam | Photo / Logo | 🟡 | — | Nestlé public website |
| F2.2 | **Illustrative Gantt Chart — một ca thực tế** | 5 dòng (R1–R5) × 480 phút. Thể hiện: batch blocks (màu theo SKU), setup gaps (màu xám), downtime windows (màu đen), pipeline events (nhỏ ở dưới). Không cần chính xác — chỉ cần minh họa mức độ phức tạp của 1 ca | Gantt chart | 🔴 | matplotlib / draw.io | Tự vẽ (illustrative) |
| T2.1 | **Bảng thông số hệ thống vật lý** | 2 cột: Parameter / Value. Bao gồm: số roasters, số GC silos/line, GC silo capacity, số RC silos/line, RC silo capacity, processing time range (13–16 min), setup time (5 min), PSC cycle time (10 min), pipeline op durations | Summary table | 🔴 | Word / LaTeX | Từ problem description |

---

### Section 2.2 — Literature Review

| ID | Tiêu đề | Mô tả nội dung | Loại | Ưu tiên | Công cụ gợi ý | Nguồn |
|---|---|---|---|---|---|---|
| T2.2 | **Literature Classification Table** | Mỗi dòng = 1 paper. Cột: Author & Year / Problem Type / Method / MTO? / MTS? / Silo or Tank Constraint? / Pipeline Mutex? / Food Context? / Similarity to This Thesis (High/Med/Low). Đây là bảng trung tâm của Section 2.2 | Comparison table | 🔴 | Word / LaTeX | Sau khi có literature list |
| F2.3 | **Literature Taxonomy Diagram** | Cây phân nhánh: Production Scheduling → (Exact Methods / Heuristics / Hybrid). Exact → (MILP / CP). Mỗi nhánh: label ví dụ papers. Highlight nhánh CP-SAT là nhánh của thesis | Tree diagram | 🟡 | draw.io / PowerPoint | Sau khi có literature list |

---

### Section 2.3 — Candidate Solution Methods

| ID | Tiêu đề | Mô tả nội dung | Loại | Ưu tiên | Công cụ gợi ý | Nguồn |
|---|---|---|---|---|---|---|
| T2.3 | **Comparison Table: MILP vs. CP-SAT vs. Metaheuristic** | 3 cột phương pháp x nhiều dòng tiêu chí: Optimality guarantee, LP relaxation quality, Disjunctive constraint handling, Setup time modeling, Variable-duration ops, Implementation effort, Scalability, Industrial precedent. Mỗi ô: checkmark / cross / Partial | Comparison table | 🔴 | Word / LaTeX | Từ formulation doc |
| F2.4 | **LP Relaxation Weakness Illustration** | Sketch đơn giản: feasible region của LP (lớn, bao gồm fractional points) vs. integer feasible set (nhỏ hơn nhiều, cách xa). Minh họa tại sao MIP branch-and-bound tốn nhiều nodes cho disjunctive problems | Conceptual diagram | 🟢 | draw.io / PowerPoint | Tự vẽ |
| F2.5 | **CP NoOverlap Propagation Illustration** | Domain reduction step-by-step: trước propagation (nhiều possible start times) → sau NoOverlap propagation (domain bị cắt). Minh họa sức mạnh của CP inference | Conceptual diagram | 🟢 | draw.io | Tự vẽ |

---

## Chapter 3 — Methodology

### Section 3.1 — Approach Comparison and Selection

| ID | Tiêu đề | Mô tả nội dung | Loại | Ưu tiên | Công cụ gợi ý | Nguồn |
|---|---|---|---|---|---|---|
| T3.1 | **Weighted Scoring Table — Method Selection** | 3 phương pháp (MILP / CP-SAT / Metaheuristic) x 5–6 tiêu chí có trọng số: Solution quality (30%), Constraint expressiveness (25%), Solve time (20%), Optimality guarantee (15%), Implementation effort (10%). Mỗi ô: điểm 1–5 x weight → tổng điểm. Kết quả: CP-SAT thắng | Weighted scoring table | 🔴 | Word / LaTeX | Tự xây dựng |

---

### Section 3.2 — Proposed Solution Design

| ID | Tiêu đề | Mô tả nội dung | Loại | Ưu tiên | Công cụ gợi ý | Nguồn |
|---|---|---|---|---|---|---|
| F3.1 | **Full System Architecture Diagram** ⭐ | Pipeline đầy đủ: [Raw Data] → [Data Preprocessing & Parsing] → [MILP Formulation via CPLEX] → [CP-SAT Formulation via OR-Tools] → [Solution Comparison Engine] → [Output: Schedule + Penalty Report]. Đây là hình quan trọng nhất của Chapter 3 | Architecture diagram | 🔴 | draw.io | Tự vẽ |
| F3.2 | **Shift Timeline Overview** | Trục ngang = 480 time slots. Đánh dấu: PSC consumption events (mỗi 10 slot, tam giác nhỏ), changeover window (block màu), downtime windows (block xám), MTO due deadline (slot 240, đường đỏ đứt). Orientation diagram cho toàn bộ model | Timeline diagram | 🔴 | matplotlib / draw.io | Tự vẽ |
| F3.3 | **GC Silo Lifecycle State Diagram** | Một silo trải qua các trạng thái: FULL (SKU-A) → [consume events] → PARTIAL (SKU-A) → [consume] → EMPTY → [reassign decision] → [replenish] → FULL (SKU-B). Kèm điều kiện chuyển trạng thái và constraint liên quan (C3, C23) | State machine | 🔴 | draw.io | Tự vẽ |
| F3.4 | **GC Pipeline Mutual Exclusion — Timeline Visualization** | Trục ngang = time slots. Một dòng = L1 pipeline. Màu blocks: xanh lam = consume (2 slots), xanh lá = replenish (2 slots), đỏ = dump (5+ slots), trắng = idle. Không có 2 màu cùng lúc. Thể hiện rõ constraint B1 | Colored timeline | 🔴 | matplotlib / draw.io | Tự vẽ |
| F3.5 | **Variable-Duration Dump Illustration** | So sánh 2 dump events: dump 300 kg → pipeline blocked 8 slots (5+3); dump 1,500 kg → pipeline blocked 20 slots (5+15). Thể hiện trade-off: dump lớn giải phóng nhiều GC hơn nhưng block pipeline lâu hơn | Annotated timeline | 🟡 | draw.io / matplotlib | Tự vẽ |
| F3.6 | **RC Silo Buffer Dynamics Chart** | Đồ thị line chart: trục Y = RC inventory level (kg), trục X = time slots. RC level tăng theo bậc thang khi batch complete, giảm theo bậc thang tại PSC consumption events (mỗi 10 slot). Vùng dưới safety stock (10,000 kg) tô màu vàng. Điểm shortage tô màu đỏ | Line chart | 🔴 | matplotlib | Tự vẽ (illustrative) hoặc từ solver output |
| F3.7 | **Model Formulation Block Structure** | Sơ đồ block không dùng ký hiệu toán: 5 khối chính: Sets & Indices → Parameters → Decision Variables → Constraint Blocks (A–H, liệt kê tên) → Objective Function (5 penalty components). Mũi tên thể hiện dependency | Block structure diagram | 🟡 | draw.io / PowerPoint | Từ math formulation doc |
| T3.2 | **Bảng tóm tắt Constraint Blocks** | Cột: Block ID / Tên block / Số ràng buộc chính / Hard hay Soft / Variable types liên quan / Mô tả ngắn. 8 blocks A–H | Summary table | 🔴 | Word / LaTeX | Từ math formulation doc |
| T3.3 | **Bảng Decision Variables** | Cột: Symbol / Domain / Description. Nhóm theo: batch scheduling vars / GC pipeline vars / GC sourcing vars / RC assignment vars / state vars / penalty tracking vars | Table | 🔴 | Word / LaTeX | Từ math formulation doc |
| T3.4 | **Bảng Parameters** | Cột: Symbol / Value / Unit / Description. Nhóm theo: roaster config / job params / BOM / capacity & thresholds / pipeline durations / PSC consumption / penalty weights | Table | 🔴 | Word / LaTeX | Từ math formulation doc |
| F3.8 | **Toy Example Gantt Chart** | Gantt chart của toy instance (60 slots, R1 + R2). Thể hiện: batch blocks (màu theo SKU), setup gaps, pipeline events ở dòng riêng bên dưới. Mục đích: concrete illustration của model behavior trước khi đi vào full instance | Gantt chart | 🟡 | matplotlib / draw.io | Từ toy example trong math formulation doc |

---

## Tổng hợp theo ưu tiên

| Ưu tiên | Figures | Tables | Tổng |
|---|---|---|---|
| 🔴 Bắt buộc | 11 | 7 | **18** |
| 🟡 Nên có | 5 | 1 | **6** |
| 🟢 Nice to have | 2 | 0 | **2** |
| **Tổng** | **18** | **8** | **26** |

---

## Công cụ gợi ý theo loại hình

| Công cụ | Phù hợp cho | Ghi chú |
|---|---|---|
| **draw.io / diagrams.net** | F1.1, F1.2, F1.3, F1.4, F1.5, F1.6, F2.3, F2.4, F2.5, F3.1, F3.2, F3.3, F3.4, F3.7, F3.8 | Miễn phí, export PNG/SVG, có template sẵn |
| **Python matplotlib** | F3.4, F3.6, F3.2 (nếu muốn chính xác), scalability curves ở Chapter 4 trở đi | Cần code nhưng reproducible và professional |
| **Word / LaTeX** | Tất cả Tables | LaTeX `booktabs` cho đẹp hơn |
| **PowerPoint / Canva** | F1.4, F1.5, F2.3, F3.1 | Nhanh hơn draw.io nếu đã quen |
| **Nestlé public website** | F2.1 | Chỉ dùng ảnh/logo public, không dùng tài liệu nội bộ |

---

## Thứ tự nên làm (theo dependency khi viết)

```
Đợt 1 — Cần có để bắt đầu viết Chapter 1:
  F1.1 → F1.2 → F1.4 → F1.5 → T1.1

Đợt 2 — Cần có để viết Chapter 2:
  T2.1 → F2.2 → T2.3
  T2.2 và F2.3 ← phụ thuộc vào literature search session (làm sau)

Đợt 3 — Cần có để viết Chapter 3:
  T3.1 → F3.1 → F3.2 → F3.3 → F3.4 → F3.6
  T3.2 → T3.3 → T3.4

Đợt 4 — Sau khi có solver output:
  F3.6 (version chính xác từ model, thay bản illustrative)
  F3.8 (từ toy instance solver output)
  Scalability curves (Chapter 4 trở đi)
```

---

*Ghi chú: F3.1 (System Architecture Diagram) ⭐ là hình quan trọng nhất của toàn bộ Chapter 3 — cần vẽ kỹ và chi tiết.*
