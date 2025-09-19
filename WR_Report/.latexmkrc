# Cấu hình latexmk để đưa file phụ vào build, file PDF vào result
$out_dir = 'build';

# Đặt tên file PDF output giống tên file tex
$jobname = '%A';

# Sử dụng pdflatex
$pdf_mode = 1;

# Hook để copy file PDF vào thư mục result sau khi build xong
$success_cmd = 'if not exist "result" mkdir "result" & copy "build\\%R.pdf" "result\\%R.pdf"';

# Tự động làm sạch file tạm khi build thành công
$cleanup_mode = 1;

# Các file cần giữ lại sau khi cleanup (chỉ giữ PDF)
$clean_ext = 'aux bbl bcf blg fdb_latexmk fls log run.xml synctex.gz';

# Tự động build khi file thay đổi (nếu muốn)
# $preview_continuous_mode = 1;

# Cài đặt biber cho bibliography
$biber = 'biber %O %S';