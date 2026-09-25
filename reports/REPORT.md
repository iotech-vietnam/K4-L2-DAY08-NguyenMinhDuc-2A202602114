# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyen Minh Duc

Công cụ gán nhãn đã dùng: CVAT (Docker local v2.74.1)

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian và có vùng đệm ở giữa thay vì chia ngẫu nhiên vì đây là dữ liệu video thu thập từ một camera giám sát giao thông có góc quay cố định. Trong video giám sát, các khung hình liền kề nhau chỉ cách nhau một phần nhỏ của giây nên có sự tương quan thời gian (temporal correlation) cực kỳ cao: vị trí, góc chiếu sáng, và bối cảnh xe cộ hầu như không đổi giữa các frame kế tiếp. 

Nếu chia ngẫu nhiên (random split), các frame liền kề của cùng một thời điểm sẽ cùng xuất hiện ở cả tập train và tập test (gây rò rỉ dữ liệu - data leakage). Khi đó, số đo trên tập kiểm thử sẽ bị **lệch lạc quan** (optimistically biased) rất nặng: mô hình chỉ cần "học vẹt" hoặc ghi nhớ vị trí cụ thể của từng chiếc xe ở khung hình lân cận là đã có thể đạt điểm AP50 rất cao, nhưng khi triển khai thực tế trên đoạn video mới thì hiệu năng sẽ sụt giảm nghiêm trọng. Việc chia theo trục thời gian kèm vùng đệm (buffer zone) đảm bảo tập test phản ánh trung thực khả năng khái quát hoá (generalization) của mô hình sang một khoảng thời gian hoàn toàn mới.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `reports/rounds_table.md`:
```
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
```

Dựa vào `outputs/compare_round0.jpg`, mô hình khởi đầu lạnh (YOLOv8n pretrained trên COCO) không khớp nhãn tham chiếu chủ yếu ở:
- Các xe ở xa có kích thước nhỏ phía trên trục đường.
- Các xe bị che khuất một phần trong bóng tối hoặc xe bị cắt mép khung hình.
- Nhận nhầm các vệt sáng phản chiếu của đèn pha trên mặt đường ướt thành box xe (false positive).

Độ phủ (Recall@0.25) phân theo kích thước xe cho thấy sự phân hóa rất rõ:
- **Xe nhỏ (R small):** chỉ đạt 0.182 (18.2%), tức mô hình bỏ sót hơn 80% xe nhỏ ở xa.
- **Xe trung bình (R medium):** đạt 0.547.
- **Xe lớn (R large):** đạt 0.561.
Điều này chứng minh trọng số pretrained từ COCO phát hiện tương đối ổn các xe ở cự ly gần và trung bình, nhưng gặp khó khăn lớn đối với xe nhỏ trong điều kiện ánh sáng ban đêm phức tạp.

Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai: Đó là các trường hợp xe ở quá xa, hình ảnh chỉ còn 2 chấm đèn sáng mờ và chiều cao box dưới 16 pixel (theo quy tắc trong `GUIDELINE_LABEL.md` và `data/DATA.md` là không bắt buộc gán nhãn), hoặc xe bị khuất quá 80%. Vì nhãn test được sinh tự động bởi một mô hình khác mà chưa qua kiểm định thủ công 100%, nên những điểm bất đồng ở rìa phân định này cần chuyên gia con người rà soát trước khi vội quy kết mô hình dự đoán sai.

## 3. Chiến lược chọn mẫu

Công thức tính điểm ưu tiên chọn mẫu:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$
(với bộ trọng số mặc định: $W_U = 0.5$, $W_A = 0.3$, $W_D = 0.2$).
- **$U$ (Uncertainty):** Độ bất định trung bình của 5 box khó nhất trong frame ($u(c) = 1 - |2c - 1|$). Khi confidence $c = 0.5$, $u$ đạt giá trị cực đại là 1 (mô hình hoàn toàn do dự không biết đó có phải là xe hay không).
- **$A$ (Ambiguity):** Tỷ lệ số box mập mờ có confidence trong khoảng $[0.15, 0.50)$ so với ngưỡng lớn nhất trong pool. Frame có $A$ cao nghĩa là có nhiều đối tượng khiến mô hình nghi ngờ.
- **$D$ (Diversity):** Khoảng cách thời gian tới frame gần nhất đã gán nhãn chia cho trần chuẩn hóa $10.0$s. Thành phần này thúc đẩy chọn các frame phân bố rải rác trên dòng thời gian, tránh tập trung cục bộ.
- **Vai trò của `MIN_GAP_S`:** Đây là ngưỡng khoảng cách thời gian tối thiểu giữa các frame được chọn trong cùng một đợt. Do camera góc nhìn cố định, các khung hình cách nhau dưới vài giây gần như là ảnh trùng lặp (near-duplicates). Nếu không có `MIN_GAP_S`, thuật toán sẽ chọn liên tiếp 5-10 frame ở cùng một cảnh tắc đường, gây lãng phí lớn chi phí gán nhãn mà mô hình không tiếp nhận thêm thông tin mới.

Dẫn chứng từ `reports/SELECTION.md`:
- `frame_0182.jpg` (Rank 1, score = 0.9591): Đứng đầu do hội tụ cả độ bất định cao ($U = 0.9182$), số box mập mờ tối đa ($A = 1.0$) và khoảng cách tối đa ($D = 1.0$).
- `frame_0331.jpg` (Rank 5, score = 0.9154): Chứa mật độ xe dày đặc (47 box dự đoán, 18 box mập mờ).
- `frame_0099.jpg` (Rank 8, score = 0.9063): Có giá trị $U = 0.9460$ cao nhất trong các frame được chọn vào vòng 1, đại diện cho đoạn đầu video ($t = 39.6$s).
- `frame_0372.jpg` (Rank 6, score = 0.9101, $t = 148.8$s): Bị loại bỏ khỏi lô chọn dù điểm rất cao, vì chỉ cách `frame_0369.jpg` ($t = 147.6$s) đúng 1.2 giây (vi phạm `MIN_GAP_S`).

Điểm bất định **không chứng minh** ảnh đó chắc chắn sẽ cải thiện mô hình. Điểm bất định chỉ phản ánh việc mô hình hiện tại đang thiếu tự tin. Nếu ảnh được chọn chứa quá nhiều nhiễu, vật thể biến dạng quá mức hoặc sau khi con người sửa nhãn làm thay đổi mạnh phân phối dự đoán khiến mô hình trở nên quá dè dặt, việc nạp thêm ảnh đó vẫn có thể làm giảm AP50 trên tập test.

## 4. Các vòng học chủ động (active learning)

Bảng tổng hợp từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 344 | 0.520 | -0.251 | 1.000 | 0.159 | 0.274 | 0.000 | 0.135 | 0.585 |

Trình bày chi tiết Vòng 1:
- **Mức độ can thiệp sửa nhãn (từ `outputs/round1_diff.md`):**
  - Đã rà soát 12 ảnh với tổng cộng 344 box hoàn chỉnh.
  - Ban đầu mô hình đề xuất 169 box: chấp nhận giữ nguyên (**accepted 138**), chỉnh sửa viền (**edited 18**), xóa box sai (**deleted 13**), và bổ sung thêm các xe AI bỏ sót (**added 188**). Tỷ lệ accept rate là 81.66%.
- **Biến động AP50:**
  - AP50 giảm từ 0.771 xuống 0.520 ($\Delta = -0.251$).
  - Tuy nhiên, độ chính xác **Precision@0.25 tăng lên tuyệt đối 1.000** (không còn phát hiện nhầm bất kỳ đối tượng giả nào), nhưng **Recall@0.25 sụt giảm từ 0.489 xuống 0.159**.
- **Biến động theo kích thước xe:**
  - Nhóm xe lớn (R large) tăng nhẹ từ 0.561 lên **0.585**.
  - Nhóm xe trung bình (R medium) giảm từ 0.547 xuống 0.135.
  - Nhóm xe nhỏ (R small) giảm về 0.000.

Đối chiếu hình ảnh và phân tích nguyên nhân:
- Dựa vào `outputs/compare_round1.jpg`, sau khi fine-tune với lô 12 ảnh được con người nới box chuẩn xác ôm sát thân xe và xóa các box vệt đèn pha, mô hình đã học được cách định vị ranh giới xe rất khắt khe. Nó chỉ tự tin đưa ra dự đoán trên những xe to, nhìn rõ nét ở tiền cảnh (giúp Precision đạt 1.0 và R large đạt 0.585), trong khi hoàn toàn e dè trước các đốm xe nhỏ ở phía xa (làm Recall của nhóm xe nhỏ rớt xuống 0).
- **Phân biệt ba góc độ:**
  1. *Quan sát độc lập (`reports/BLIND_SCAN.md`):* Trên `frame_0099.jpg`, mắt người nhận diện 26 xe và chỉ ra điểm yếu của AI ở góc tối phía trên bên trái và góc mờ mép dưới.
  2. *Lỗi pre-label đã sửa (`outputs/round1_diff.md` & `reports/REVIEW_LOG.csv`):* AI ban đầu chỉ phát hiện 13 xe (bỏ sót 50%), bắt nhầm vệt đèn (`frame_0182.jpg`); người gán đã bổ sung đủ 26 xe và cắt gọt viền box sát thân xe theo đúng quy tắc.
  3. *Kết quả mô hình sau train:* Mô hình fine-tune đã khắc phục hoàn toàn lỗi nhận diện giả (không còn vẽ box lên vệt đèn phản chiếu), nhưng vì dữ liệu huấn luyện còn nhỏ (chỉ 12 ảnh) nên mô hình bị hội tụ về xu hướng dự đoán thận trọng cao độ.
- **Ca khó theo guideline:** Trường hợp xe bị xe tải phía trước che khuất hơn một nửa thân xe trên `frame_0099.jpg` và `frame_0331.jpg`. Theo guideline, người gán chỉ vẽ box ôm phần nhìn thấy của xe, nhưng mô hình sau khi train vẫn chưa đủ lượng mẫu đa dạng để học được biểu diễn của các xe bị che khuất nặng này.

## 5. Kết luận và giới hạn

So với cold start, mô hình sau vòng 1 có sự đánh đổi rõ rệt: cải thiện vượt trội về độ tinh sạch dự đoán (Precision đạt 100%, loại bỏ triệt để các box giả trên mặt đường và tăng Recall cho xe lớn), nhưng lại làm sụt giảm độ bao phủ chung (AP50 giảm do Recall các xe nhỏ ở xa bị hạ thấp). Quyết định dừng lại ở vòng 1 là phù hợp với phạm vi bài lab để tiến hành đánh giá, rút ra bài học về hiện tượng sụt giảm Recall khi fine-tune trên lô nhỏ với nhãn gán nghiêm ngặt.

Đề xuất hai ca còn yếu hoặc bất định cho vòng tiếp theo (từ `outputs/selection_round2.csv`):
1. `frame_0002.jpg` ($t = 0.8$s, Rank 1 trong Vòng 2): Giúp bổ sung mẫu ở giai đoạn đầu video với mật độ xe nhỏ đa dạng.
2. `frame_0067.jpg` ($t = 26.8$s, Rank 4 trong Vòng 2): Bổ sung thêm bối cảnh trung gian về thời gian, hỗ trợ mô hình nhận diện lại các xe cỡ nhỏ và trung bình mà không sợ vi phạm khoảng cách thời gian (`MIN_GAP_S`).
*Đánh giá chi phí:* Mỗi frame này có từ 25 - 35 xe, đòi hỏi thời gian rà soát khoảng 3 - 5 phút/ảnh, nhưng mang lại giá trị cao vì nằm ở các khoảng thời gian chưa từng xuất hiện trong tập train vòng 1.

Các giới hạn ảnh hưởng đến kết luận:
- Tập kiểm thử chỉ có 20 ảnh và nhãn tham chiếu do một mô hình tạo tự động, chưa được con người rà duyệt hoàn toàn. Các box nhỏ dưới 16 pixel bị bỏ qua theo luật, nhưng các box nhấp nhô quanh ngưỡng 16 - 20 pixel có thể tạo ra độ lệch lớn trong phép đo Recall.
- Số lượng ảnh huấn luyện 12 ảnh là quá nhỏ so với độ phức tạp của bài toán phát hiện vật thể ban đêm, khiến mô hình dễ bị trôi trọng số (catastrophic drift) so với nền tảng pretrained COCO ban đầu.

Nếu AP50 giảm, ba điều cần kiểm tra trước khi train thêm:
1. **Kiểm tra phân phối Confidence Score:** Xem các box dự đoán của mô hình có thực sự biến mất hay chỉ bị tụt confidence xuống dưới ngưỡng $0.25$.
2. **Kiểm tra tính nhất quán (Consistency) của nhãn gán:** Đảm bảo tất cả các ảnh trong các vòng đều tuân thủ chặt chẽ một quy chuẩn duy nhất (ví dụ không bao giờ tính vệt đèn, luôn ôm sát mép thân).
3. **Điều chỉnh Hyperparameters khi Fine-tune:** Giảm learning rate, sử dụng freeze backbone hoặc tăng trọng số loss phân loại/bounding box để giữ lại các đặc trưng tổng quát của mô hình COCO gốc, tránh việc mô hình trở nên quá cực đoan chỉ sau 1 lô huấn luyện nhỏ.
