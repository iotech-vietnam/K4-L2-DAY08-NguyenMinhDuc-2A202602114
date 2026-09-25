# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét ảnh gần trùng hoặc trường hợp model không dự đoán được box:
Nếu chỉ có ngân sách rà 5 ảnh, 5 frame được ưu tiên chọn là:
1. `frame_0182.jpg` (Rank 1, điểm score = 0.9591, thời điểm t = 72.8s): Đạt điểm cao nhất toàn pool, hệ số box mập mờ A = 1.0 (18 box lưỡng lự) và độ bất định U = 0.9182, cung cấp nhiều thông tin về các ca khó phân định.
2. `frame_0369.jpg` (Rank 2, điểm score = 0.9324, thời điểm t = 147.6s): Mật độ giao thông ban đêm dày đặc (43 box dự đoán, 16 box mập mờ), nhiều xe bị che khuất một phần.
3. `frame_0099.jpg` (Rank 8, điểm score = 0.9063, thời điểm t = 39.6s): Phân bổ ở thời điểm đầu video (t = 39.6s), có độ bất định box cao nhất nhóm đầu (U = 0.9460), phản ánh ca xe bị che khuất ở mép và góc tối.
4. `frame_0227.jpg` (Rank 11, điểm score = 0.8915, thời điểm t = 90.8s): Thời điểm giữa video (t = 90.8s, cách xa các frame trên > 15s), đảm bảo tính đa dạng mẫu thời gian (D = 1.0).
5. `frame_0326.jpg` (Rank 4, điểm score = 0.9155, thời điểm t = 130.4s): Có 39 box và 15 box mập mờ, độ bất định U = 0.9310.
Quyết định loại trừ ảnh gần trùng: Không chọn `frame_0372.jpg` (Rank 6, score = 0.9101, t = 148.8s) dù điểm rất cao, vì nó chỉ cách `frame_0369.jpg` đúng 1.2 giây; do camera góc nhìn cố định, hai frame này là ảnh gần trùng (near-duplicate), gán cả hai sẽ lãng phí ngân sách mà không bổ sung tri thức mới.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
1. `frame_0182.jpg` (Rank 1 trong CSV, score 0.9591, U = 0.9182, A = 1.0): Trên contact sheet hiển thị nhiều vùng phản chiếu ánh đèn đường và các box có confidence dao động quanh 0.3 - 0.5.
2. `frame_0331.jpg` (Rank 5 trong CSV, score 0.9154, n_boxes = 47, n_ambiguous = 18): Là frame có số lượng xe đông nhất trong lô được chọn, nhiều cụm xe đi sát nhau ở các làn đường xa.
3. `frame_0099.jpg` (Rank 8 trong CSV, score 0.9063, U = 0.9460, n_ambiguous = 14): Frame có giá trị U cao nhất trong các frame được chọn vào lô vòng 1, trên contact sheet cho thấy rõ các ca xe mép ảnh bị cắt và xe tối góc trên bên trái.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- `frame_0372.jpg` (Rank 6, score = 0.9101, U = 0.9202, A = 0.8333): Có điểm cao hơn cả frame 0312, 0099, 0187,... nhưng bị thuật toán loại bỏ khỏi lô chọn vì vi phạm ràng buộc `MIN_GAP_S` (chỉ cách `frame_0369.jpg` 1.2s). Nếu chọn thủ công có hạn chế ngân sách, ta cũng chủ động bỏ qua frame này để tránh trùng lặp thông tin.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
Phép chọn bằng độ bất định (Uncertainty Sampling) chỉ đo lường sự do dự/lúng túng của mô hình hiện tại (confidence gần ngưỡng 0.5), chứ không chứng minh được rằng sau khi gán nhãn và train các ảnh này thì chất lượng mô hình chắc chắn sẽ tăng. Nếu dữ liệu train quá ít (chỉ 12 ảnh) hoặc nhãn sau khi sửa làm thay đổi phân phối dự đoán khiến mô hình trở nên quá thận trọng (tăng Precision lên 1.0 nhưng Recall sụt giảm), điểm AP50 trên tập test vẫn có thể giảm sút.
