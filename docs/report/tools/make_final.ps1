
$dir="D:\lap_trinh\python\slidegen-rag\docs\report"
$src="C:\Users\tuanq\Downloads\HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG.docx"
Copy-Item $src "$dir\_cv.docx" -Force
$w=New-Object -ComObject Word.Application; $w.Visible=$false
$d=$w.Documents.Open("$dir\_cv.docx")
function SetT($old,$new){ $r=$d.Content; $f=$r.Find; $f.ClearFormatting(); $f.Replacement.ClearFormatting()
  $ok=$f.Execute($old,$true,$false,$false,$false,$false,$true,0,$false,$new,2); "$ok" }
SetT "VIỆN KINH TẾ BƯU ĐIỆN" "KHOA CÔNG NGHỆ THÔNG TIN 1"
SetT "BỘ MÔN PHÁT TRIỂN KỸ NĂNG" "BỘ MÔN PYTHON"
SetT "BÀI THI CUỐI KỲ" "BÀI TẬP LỚN"
SetT "HỌC PHẦN: KỸ NĂNG LÀM VIỆC NHÓM" "HỌC PHẦN: LẬP TRÌNH PYTHON"
SetT "Sản phẩm: Tương lai của sinh viên it trong thời đại AI" "Đề tài: HRAG – Hierarchical Agentic RAG, hệ thống hỏi đáp tài liệu dài"
$s=$d.Sections.Item(1).PageSetup; $s.PageWidth=595.3; $s.PageHeight=841.9
$r=$d.Content; $r.Collapse(0); $r.InsertBreak(2)
$r=$d.Content; $r.Collapse(0); $r.InsertFile("$dir\_body.docx")
$p=$d.Sections.Item(2).PageSetup
$p.TopMargin=70.9; $p.BottomMargin=70.9; $p.LeftMargin=85.05; $p.RightMargin=56.7
function HS($name,$size,$color,$ital){ $st=$d.Styles.Item($name); $st.Font.Name="Times New Roman"; $st.Font.Size=$size; $st.Font.Bold=$true; $st.Font.Italic=$ital; $st.Font.Color=$color
  $pf=$st.ParagraphFormat; $pf.LeftIndent=0; $pf.FirstLineIndent=0; $pf.Alignment=0; $pf.SpaceBefore=12; $pf.SpaceAfter=6; $pf.KeepWithNext=$true; $pf.KeepTogether=$true }
HS "Heading 1" 16 6567967 $false
HS "Heading 2" 14 6567967 $false
HS "Heading 3" 13 0 $true
$d.TablesOfContents.Item(1).Update()
"pages: " + $d.ComputeStatistics(2)
$d.SaveAs2("$dir\BaoCao_slidegen-rag.docx",16)
$d.ExportAsFixedFormat("$dir\BaoCao_slidegen-rag.pdf",17)
$d.Close($false); $w.Quit()
