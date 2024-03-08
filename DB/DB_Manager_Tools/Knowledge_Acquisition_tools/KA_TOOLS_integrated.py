from DB.DB_Manager_Tools.Knowledge_Acquisition_tools.Literature_Search import PdfMatch
from DB.DB_Manager_Tools.Knowledge_Acquisition_tools.Pdf_Convert_txt import PdfConvert
from DB.DB_Manager_Tools.Knowledge_Acquisition_tools.Txt_embedding import Txtembedding


PdfMatch = PdfMatch()
PdfConvert = PdfConvert()
Txtembedding = Txtembedding()
ka_tools_list = [PdfMatch, PdfConvert, Txtembedding, ]
