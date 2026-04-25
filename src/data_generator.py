
# ============================================================
#  src/data_generator.py  -- Sinh du lieu huan luyen tong hop
#  Phien ban PhoBERT: format don gian {text, label, label_id, entities}
#  Bao gom: Zipfian distribution, Data Augmentation đa dạng
# ============================================================

import json, random, re, os, unicodedata, time, math
from src.config import (
    LABEL2ID, DATA_PROCESSED, INTENT_LABELS, FINAL_ENTITY_LABELS,
    FRAME_SCHEMA, ALL_SLOT_NAMES, SLOT2IDX, NUM_SLOTS, SLOT_TO_ENTITY,
    GENRE_ALIASES,
)

try:
    from google import genai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# API key chi lay tu bien moi truong, KHONG hardcode
LLM_API_KEY = os.environ.get("GEMINI_API_KEY")
if LLM_AVAILABLE and LLM_API_KEY:
    llm_client = genai.Client(api_key=LLM_API_KEY)
else:
    llm_client = None
    if not LLM_API_KEY:
        print("[data_generator] GEMINI_API_KEY chua duoc set -> LLM augmentation TAT")

# Đã thêm dấu Tiếng Việt chuẩn
TEMPLATES = {
    "find_movie": [
        "Tôi muốn tìm phim {MOVIE_TITLE}",
        "Cho tôi xem {MOVIE_TITLE}",
        "Tìm kiếm phim {MOVIE_TITLE}",
        "Bạn có phim {MOVIE_TITLE} không?",
        "Tìm giúp tôi bộ phim {MOVIE_TITLE}",
        "Tìm phim {GENRE} của {ACTOR} năm {YEAR}",
        "Phim {GENRE} hay của {ACTOR}",
        "Có phim nào của {ACTOR} năm {YEAR} không?",
        "{ACTOR} có phim gì năm {YEAR}",
        "Phim {GENRE} do {DIRECTOR} đạo diễn",
        "{ACTOR} đóng phim {GENRE} nào vậy?",
        "Năm {YEAR} {ACTOR} có phim gì không?",
        "Tìm phim {GENRE} mà {ACTOR} đóng chính",
        "{DIRECTOR} đạo diễn phim {GENRE} gì không?",
        "Phim {GENRE} của {DIRECTOR} năm {YEAR}",
        "Tìm phim tên là {MOVIE_TITLE} giúp tôi với",
        "Có cách nào xem {MOVIE_TITLE} không?",
        "Tôi nhớ mang máng phim {MOVIE_TITLE}, tìm giúp tôi",
        "Tra cứu bộ phim {MOVIE_TITLE} cho tôi",
        "Làm sao để xem phim {MOVIE_TITLE} vậy?",
        "Phim {GENRE} nào có {ACTOR} đóng trong đó không?",
        "Tôi đang tìm phim {GENRE} ra mắt năm {YEAR}",
        "Có bộ phim nào mà {ACTOR} đóng chính năm {YEAR} không?",
        "Phim do {DIRECTOR} đạo diễn theo thể loại {GENRE} là gì?",
        "Tìm cho tôi phim {GENRE} có {ACTOR} tham gia năm {YEAR}",
        "Tìm cho tôi bộ phim mà {ACTOR} đóng vai phản diện",
        "Phim nào có {ACTOR} và {DIRECTOR} cùng hợp tác năm {YEAR}?",
        "Tôi nghe nói có phim {GENRE} rất hay, tìm giúp tôi",
        "Bộ phim {MOVIE_TITLE} hiện có trên nền tảng nào?",
        "Tìm phim {GENRE} có rating cao nhất năm {YEAR}",
        "Phim {GENRE} nào dài trên 2 tiếng của {DIRECTOR}?",
        "Có bản remake nào của {MOVIE_TITLE} không?",
        "Tìm phim có cốt truyện tương tự {MOVIE_TITLE}",
        "Phim nào mà cả {ACTOR} lẫn {DIRECTOR} đều tham gia?",
        "Tôi muốn tìm phần 2 của {MOVIE_TITLE}",
        "Bộ phim {GENRE} nào được chuyển thể từ tiểu thuyết?",
        "Tìm phim {GENRE} đoạt giải Oscar năm {YEAR}",
        "Có phim nào của {ACTOR} chưa được chiếu ở Việt Nam không?",
        "Phim {MOVIE_TITLE} có bản director's cut không?",
        "Tìm những bộ phim {GENRE} kinh phí thấp nhưng chất lượng cao",
        "Phim nào của {DIRECTOR} được ra mắt gần đây nhất?",
        "Tìm phim {GENRE} sản xuất bởi {DIRECTOR} trước năm {YEAR}",
        "Bộ phim nào có {ACTOR} đóng vai chính lần đầu tiên?",
        "Tôi quên tên phim rồi, chỉ nhớ {ACTOR} đóng thể loại {GENRE}",
        "Phim {GENRE} nào của {ACTOR} chưa ra mắt Việt Nam năm {YEAR}?",
        # --- Pattern "có {ACTOR}" ---
        "Cho tôi xem phim {GENRE} có {ACTOR}",
        "Phim {GENRE} có {ACTOR} đóng",
        "Có phim {GENRE} nào có {ACTOR} không?",
        "Tìm phim {GENRE} mà có {ACTOR} tham gia",
        "Phim {GENRE} nào mà có {ACTOR} đóng chính vậy?",
        "Cho tôi phim có {ACTOR} thuộc thể loại {GENRE}",
        "Tìm phim {GENRE} có {ACTOR} đóng năm {YEAR}",
        "Phim nào có {ACTOR} đóng mà thuộc dòng {GENRE}?",
        # --- Gen Z / teencode templates ---
        "Kiếm phim {GENRE} có {ACTOR} đóng đi b",
        "Ê có phim {GENRE} nào của {ACTOR} ko?",
        "Tìm film {GENRE} có {ACTOR} xem với",
        "Cho xem phim {GENRE} mà có {ACTOR} tham gia nha",
        "Kiếm phim hay của {ACTOR} đi bn",
        "Có film nào tên {MOVIE_TITLE} ko b?",
        "Tìm giúp mk phim {GENRE} năm {YEAR} với",
        "Ê tìm phim {MOVIE_TITLE} cho t đi",
        # --- Hard negatives: find_movie vs genre_filter ---
        "Tìm phim {GENRE} của {ACTOR} cho tôi xem",
        "Tìm phim {GENRE} mà {ACTOR} đóng cùng {DIRECTOR}",
        "Cho tôi xem phim {GENRE} năm {YEAR} có {ACTOR} đóng",
        "Phim {GENRE} nào có {ACTOR} và ra năm {YEAR}?",
        "Tìm phim {GENRE} do {DIRECTOR} làm có {ACTOR} tham gia",
    ],
    "recommendation": [
        "Gợi ý cho tôi phim {GENRE} hay",
        "Đề xuất phim {GENRE} năm {YEAR}",
        "Tôi muốn xem phim {GENRE}, bạn gợi ý gì?",
        "Cho tôi một số phim {GENRE} hay nhất",
        "Phim nào hay tương tự {MOVIE_TITLE}?",
        "Recommend phim {GENRE} hay đi",
        "Tôi đang không biết xem phim gì, gợi ý đi",
        "Phim {GENRE} nào đáng xem nhất {YEAR}?",
        "Top phim {GENRE} năm {YEAR} là gì?",
        "Tôi muốn xem gì đó giống {MOVIE_TITLE}",
        "Cho tôi vài gợi ý phim {GENRE} của đạo diễn {DIRECTOR}",
        "Phim {GENRE} nào của {ACTOR} được đánh giá tốt nhất?",
        "Hôm nay tôi muốn thư giãn với phim {GENRE}, gợi ý giúp tôi",
        "Tôi vừa xem xong {MOVIE_TITLE}, có phim nào tương tự không?",
        "Bạn nghĩ phim {GENRE} nào đáng xem nhất hiện tại?",
        "Đề xuất cho tôi vài bộ phim {GENRE} chất lượng cao năm {YEAR}",
        "Nếu thích {MOVIE_TITLE} thì tôi nên xem thêm phim gì?",
        "Gợi ý phim {GENRE} phù hợp để xem cuối tuần",
        "Phim {GENRE} của {ACTOR} cái nào xem được nhất?",
        "Tôi chưa biết chọn phim gì, bạn suggest giúp tôi với",
        "Cho tôi danh sách phim {GENRE} được yêu thích nhất {YEAR}",
        "Có phim {GENRE} nào do {DIRECTOR} làm mà hay không?",
        "Tôi vừa xem hết series {MOVIE_TITLE}, giờ xem gì tiếp?",
        "Gợi ý phim {GENRE} phù hợp xem một mình ban đêm",
        "Tôi muốn xem phim để học tiếng Anh, loại {GENRE} nào tốt?",
        "Recommend cho tôi phim {GENRE} không quá dài, dưới 2 tiếng",
        "Phim {GENRE} nào phù hợp để xem với bạn gái?",
        "Tôi đang buồn, gợi ý phim {GENRE} để giải khuây",
        "Gợi ý phim {GENRE} có kết thúc happy ending",
        "Có phim nào kiểu như {MOVIE_TITLE} nhưng hay hơn không?",
        "Phim {GENRE} nào của {ACTOR} đáng xem nhất trong 5 năm qua?",
        "Tôi thích phim có twist bất ngờ, gợi ý {GENRE} đi",
        "Recommend phim {GENRE} phù hợp cho người mới bắt đầu xem thể loại này",
        "Gợi ý phim {GENRE} ít người biết nhưng rất hay",
        "Tôi muốn xem phim {GENRE} dựa trên câu chuyện có thật",
        # --- Hard negatives: genre_filter vs recommendation (no asking for suggestion) ---
        "Phim {GENRE} nào của {DIRECTOR} được đánh giá cao nhất?",
        "Suggest cho tôi phim {GENRE} có diễn xuất xuất sắc",
        "Tôi muốn xem phim {GENRE} kinh điển của thập niên 90",
        "Gợi ý phim {GENRE} phù hợp cho cả nhà cùng xem tối nay",
        "Có phim {GENRE} nào mới ra mắt {YEAR} đáng xem không?",
        "Phim {GENRE} nào có thể xem đi xem lại nhiều lần?",
        # --- Real user data patterns ---
        "Tìm phim hành động hay năm 2023",
        "Thông tin diễn viên Tom Cruise",
        "Gợi ý cho tôi top 5 phim {GENRE} hay nhất mọi thời đại",
        # --- Vague/emotional recommendation (no explicit genre/movie) ---
        "Cho tôi xem gì đó vui vui",
        "Tôi muốn xem gì đó nhẹ nhàng",
        "Tôi buồn quá, cho tôi xem phim gì đó đi",
        "Tối nay rảnh, xem phim gì hay?",
        "Tôi chán quá, gợi ý phim đi",
        "Cho xem gì đó giải trí đi",
        "Có phim nào xem cho vui không?",
        "Tôi đang rảnh, recommend phim gì đi",
        "Xem phim gì bây giờ hay ta?",
        "Muốn xem phim mà không biết chọn gì",
        "Gợi ý phim gì đó hay hay đi",
        "Cho tôi cái gì đó xem cho đỡ buồn",
        "Tối nay coi gì bây giờ?",
        "Recommend phim gì chill chill đi",
        "Suggest mấy phim hay hay đi bạn",
        # --- Gen Z recommendation ---
        "Gợi ý film hay đi bn",
        "Cho t xem phim gì đó đi",
        "Recommend phim chill đi b",
        "Có film nào xịn xịn ko?",
        "Tui muốn coi phim hay, suggest đi",
        # --- Hard negatives: recommendation vs genre_filter ---
        "Gợi ý phim {GENRE} hay nhất cho tôi xem",
        "Cho tôi phim {GENRE} nào hay để xem cuối tuần",
        "Recommend phim {GENRE} đáng xem nhất hiện tại",
        "Bạn nghĩ phim {GENRE} nào đáng xem nhất?",
        "Đề xuất cho tôi phim {GENRE} hay ho",
    ],
    "movie_info": [
        "Phim {MOVIE_TITLE} có nội dung gì?",
        "Cho tôi biết thông tin về phim {MOVIE_TITLE}",
        "{MOVIE_TITLE} ra mắt năm bao nhiêu?",
        "Điểm đánh giá của {MOVIE_TITLE} là bao nhiêu?",
        "Ai đạo diễn phim {MOVIE_TITLE}?",
        "Tóm tắt nội dung phim {MOVIE_TITLE} cho tôi",
        "Phim {MOVIE_TITLE} thuộc thể loại gì?",
        "{MOVIE_TITLE} có được đánh giá tốt không?",
        "Ai đóng vai chính trong phim {MOVIE_TITLE}?",
        "Cốt truyện của {MOVIE_TITLE} là gì?",
        "{MOVIE_TITLE} kể về chuyện gì vậy?",
        "Điểm IMDb của {MOVIE_TITLE} là bao nhiêu?",
        "Phim {MOVIE_TITLE} dài bao nhiêu tiếng vậy?",
        "Ngân sách sản xuất của {MOVIE_TITLE} là bao nhiêu?",
        "Phim {MOVIE_TITLE} có phần tiếp theo chưa?",
        "{MOVIE_TITLE} được quay ở đâu vậy?",
        "Phim {MOVIE_TITLE} có lời thoại tiếng Việt không?",
        "Nhà sản xuất của {MOVIE_TITLE} là ai?",
        "{MOVIE_TITLE} có đoạt giải thưởng nào không?",
        "Phim {MOVIE_TITLE} có phù hợp cho trẻ em xem không?",
        "Dàn diễn viên đầy đủ của phim {MOVIE_TITLE} gồm những ai?",
        "Phim {MOVIE_TITLE} dựa trên câu chuyện có thật không?",
        "Phim {MOVIE_TITLE} được quay trong bao lâu?",
        "Soundtrack của phim {MOVIE_TITLE} do ai sáng tác?",
        "Phim {MOVIE_TITLE} có bao nhiêu phần rồi?",
        "Hiệu ứng hình ảnh trong {MOVIE_TITLE} có tốt không?",
        "Phim {MOVIE_TITLE} được phát hành ở bao nhiêu quốc gia?",
        "Kịch bản phim {MOVIE_TITLE} do ai viết?",
        "Phim {MOVIE_TITLE} có tựa đề tiếng Việt là gì?",
        "Doanh thu phòng vé của {MOVIE_TITLE} là bao nhiêu?",
        "Bối cảnh chính của phim {MOVIE_TITLE} diễn ra ở đâu?",
        "Phim {MOVIE_TITLE} thuộc hãng sản xuất nào?",
        "Cảnh quay nào trong {MOVIE_TITLE} tốn nhiều kinh phí nhất?",
        "{MOVIE_TITLE} có phiên bản lồng tiếng Việt không?",
        "Phim {MOVIE_TITLE} gây tranh cãi ở điểm gì?",
        "Thông điệp chính mà {MOVIE_TITLE} muốn truyền tải là gì?",
        "{MOVIE_TITLE} có được chuyển thể thành series không?",
        "Phim {MOVIE_TITLE} phù hợp cho lứa tuổi nào?",
        "Hậu trường phim {MOVIE_TITLE} có điều gì thú vị?",
        "Phim {MOVIE_TITLE} có mid-credit hoặc post-credit scene không?",
        "{MOVIE_TITLE} nhận được phản hồi như thế nào từ khán giả?",
        "Phim {MOVIE_TITLE} được đề cử những giải gì?",
        # --- "về" pattern for movie_info ---
        "Cho tôi biết thêm về phim {MOVIE_TITLE}",
        "Cho tôi biết về {MOVIE_TITLE}",
        "Kể cho tôi nghe về phim {MOVIE_TITLE}",
        "Thông tin về phim {MOVIE_TITLE}",
        "Nói cho tôi biết về {MOVIE_TITLE} đi",
        # --- Gen Z movie_info ---
        "Phim {MOVIE_TITLE} nội dung j vậy?",
        "Cho t biết về phim {MOVIE_TITLE} đi",
        "{MOVIE_TITLE} hay ko b?",
        "Review phim {MOVIE_TITLE} cho t nghe đi",
    ],
    "person_info": [
        "{ACTOR} có những phim nào?",
        "Cho tôi danh sách phim của {ACTOR}",
        "{ACTOR} là diễn viên phim nào?",
        "Phim mới nhất của {ACTOR} là gì?",
        "{ACTOR} nổi tiếng với bộ phim nào?",
        "Kể cho tôi nghe về diễn viên {ACTOR}",
        "{ACTOR} đã đóng bao nhiêu bộ phim?",
        "Phim hay nhất của {ACTOR} là gì?",
        "{ACTOR} có hợp tác với đạo diễn {DIRECTOR} không?",
        "Diễn viên {ACTOR} đang đóng phim gì?",
        "{ACTOR} bắt đầu sự nghiệp diễn xuất từ khi nào?",
        "Vai diễn nổi bật nhất trong sự nghiệp của {ACTOR} là gì?",
        "{ACTOR} có được đề cử giải thưởng nào chưa?",
        "Gần đây {ACTOR} đang tham gia dự án phim nào?",
        "{ACTOR} thường đóng vai loại nhân vật như thế nào?",
        "Có diễn viên nào từng đóng phim cùng {ACTOR} nhiều lần không?",
        "{ACTOR} đã từng hợp tác với {DIRECTOR} trong phim nào chưa?",
        "Phim đầu tay của {ACTOR} là bộ phim nào?",
        "Xem phim nào để hiểu rõ hơn về tài năng diễn xuất của {ACTOR}?",
        "{ACTOR} ngoài đóng phim còn làm gì trong ngành giải trí?",
        "{ACTOR} học diễn xuất ở đâu vậy?",
        "Cuộc sống đời tư của {ACTOR} như thế nào?",
        "{ACTOR} có tự thực hiện cảnh hành động không?",
        "Thù lao của {ACTOR} cho một bộ phim là bao nhiêu?",
        "{ACTOR} có theo học trường diễn xuất nào không?",
        "Vai diễn nào khiến {ACTOR} nổi tiếng toàn cầu?",
        "{ACTOR} đã từng từ chối vai diễn nào đáng tiếc chưa?",
        "Phong cách diễn xuất đặc trưng của {ACTOR} là gì?",
        "{ACTOR} có làm đạo diễn hoặc sản xuất phim không?",
        "Lần đầu tiên {ACTOR} xuất hiện trên màn ảnh là bao giờ?",
        "{ACTOR} có bao nhiêu giải thưởng diễn xuất trong sự nghiệp?",
        "Dự án tiếp theo của {ACTOR} là gì?",
        "{ACTOR} thường chuẩn bị cho vai diễn như thế nào?",
        "Ai là người đã phát hiện ra tài năng của {ACTOR}?",
        "{ACTOR} có hợp tác với diễn viên nào thường xuyên nhất?",
        "Ngoài diễn xuất, {ACTOR} còn có tài năng gì khác?",
        "{ACTOR} từng đóng phim của đạo diễn {DIRECTOR} bao nhiêu lần?",
        "Phim nào đánh dấu bước ngoặt sự nghiệp của {ACTOR}?",
        "{ACTOR} có ảnh hưởng như thế nào đến thế hệ diễn viên trẻ?",
        "Vai diễn nào khó nhất mà {ACTOR} từng thực hiện?",
    ],
    "genre_filter": [
        "Cho tôi xem phim {GENRE}",
        "Danh sách phim {GENRE}",
        "Top phim {GENRE} hay nhất",
        "Phim {GENRE} chiếu rạp năm {YEAR}",
        "Tìm phim {GENRE}",
        "Có những phim {GENRE} nào hay?",
        "Tôi muốn xem phim thuộc thể loại {GENRE}",
        "Lọc phim theo thể loại {GENRE}",
        "Phim {GENRE} nào đang hot nhất?",
        "Phim {GENRE} năm {YEAR} có gì hay?",
        "Cho tôi thấy các bộ phim {GENRE} mới nhất",
        "Tôi chỉ thích xem phim {GENRE}, có gì hay không?",
        "Dạo này phim {GENRE} nào đang được chú ý?",
        "Tổng hợp phim {GENRE} hay nhất năm {YEAR} cho tôi",
        "Phim {GENRE} nào mới ra mắt gần đây?",
        "Bạn có thể lọc cho tôi danh sách phim {GENRE} không?",
        "Cho tôi xem những phim {GENRE} kinh điển nhất",
        "Phim {GENRE} nào phù hợp để xem cùng gia đình?",
        "Hiện tại phim {GENRE} nào đang chiếu rạp?",
        "Tôi muốn khám phá thể loại phim {GENRE}, nên bắt đầu từ đâu?",
        "So sánh các bộ phim {GENRE} nổi bật năm {YEAR} giúp tôi",
        "Phim {GENRE} nào có cốt truyện phức tạp và nhiều tầng lớp?",
        "Tôi muốn khám phá phim {GENRE} của điện ảnh châu Á",
        "Liệt kê phim {GENRE} có điểm Rotten Tomatoes trên 90%",
        "Phim {GENRE} nào phù hợp để xem trong ngày mưa?",
        "Gợi ý phim {GENRE} ít thoại, nhiều hình ảnh ấn tượng",
        "Phim {GENRE} nào đang trending trên mạng xã hội hiện tại?",
        "Tôi muốn xem phim {GENRE} sản xuất ngoài Hollywood",
        "Phim {GENRE} nào có soundtrack hay nhất?",
        "Cho tôi danh sách phim {GENRE} đoạt nhiều giải thưởng nhất",
        "Phim {GENRE} nào năm {YEAR} bị đánh giá thấp một cách oan uổng?",
        "Tìm phim {GENRE} có thời lượng dưới 90 phút",
        "Phim {GENRE} nào được chuyển thể từ truyện tranh hoặc game?",
        "Cho tôi xem phim {GENRE} sản xuất tại Việt Nam",
        "Phim {GENRE} nào thích hợp xem theo nhóm bạn?",
        "Danh sách phim {GENRE} có kết thúc mở, để lại nhiều suy ngẫm",
        "Phim {GENRE} nào có nhân vật phản diện được xây dựng tốt nhất?",
        "Tôi muốn xem phim {GENRE} dựa trên sự kiện lịch sử",
        "Phim {GENRE} nào năm {YEAR} gây bão phòng vé toàn cầu?",
        "Gợi ý phim {GENRE} phong cách indie, không mainstream",
        "Phim {GENRE} nào có tỷ lệ xem lại cao nhất của {YEAR}?",
    ],
    "greeting": [
        "Xin chào", "Chào bạn", "Hello", "Hi bạn",
        "Chào buổi sáng", "Hey", "Bắt đầu thôi nào",
        "Alo chatbot ơi", "Chào chatbot",
        "Mình cần tìm phim, chào bạn",
        "Chatbot ơi, cho mình hỏi",
        "Xin chào, bạn có thể giúp mình không?",
        "Bắt đầu tìm kiếm phim nào",
        "Chào buổi tối", "Chào buổi chiều",
        "Hi chatbot", "Yo", "Ê bạn ơi",
        "Chào nhé, mình cần tư vấn phim",
        "Mình mới vào, chào bạn",
        "Hello, bạn giúp mình tìm phim nhé",
        "Chào, mình muốn hỏi về phim",
        "Bot ơi", "Ê bot", "Hey bot",
        "Lần đầu mình dùng, xin chào",
        "Chào bạn, mình đang rảnh muốn xem phim",
        "Hellu", "Hế lô",
        "Chào bạn chatbot nhé",
        "Hi, tư vấn phim giúp mình",
        "Chào nha", "Xin chào bạn nhé",
        "Mình chào bạn", "Chào bạn nha",
        "Hello bạn ơi", "Hi hi",
        "Chào, giúp mình với",
        "Hey bạn, mình cần gợi ý phim",
        "Xin chào, mình muốn tìm phim hay",
        "Chào buổi trưa", "Ơi chatbot",
        "Chào bạn, hôm nay mình muốn xem phim",
    ],
    "goodbye": [
        "Tạm biệt", "Bye bye", "Cảm ơn nhé", "Thôi mình đi đây",
        "Hẹn gặp lại", "OK xong rồi cảm ơn",
        "Cảm ơn bạn nhiều lắm, tạm biệt",
        "Mình tìm được rồi, cảm ơn",
        "Thôi tắt máy đây, bye",
        "Oke cảm ơn, thoát nhé",
        "Bye nha", "Tạm biệt bạn",
        "Cảm ơn nhiều, bye", "Chào nhé, tạm biệt",
        "OK mình đi đây", "Xong rồi, cảm ơn bạn",
        "Thôi bye", "Hẹn gặp lại nhé",
        "Cảm ơn bot nhiều lắm", "Mình xong rồi, bye bye",
        "Tạm biệt chatbot", "OK mình off đây",
        "Thanks nhé", "Thank you, tạm biệt",
        "Cảm ơn bạn, hẹn gặp lại",
        "Mình đi ngủ đây, bye", "Tốt lắm, cảm ơn",
        "Bye thôi", "OK đủ rồi, cảm ơn",
        "Mình hài lòng rồi, tạm biệt",
        "Chào tạm biệt nhé", "Thoát", "Kết thúc",
        "Mình không cần nữa, cảm ơn",
        "Đủ rồi, bye bye", "Cảm ơn bạn rất nhiều",
    ],
    "out_of_scope": [
        "Hôm nay thời tiết thế nào?",
        "Bạn là AI gì vậy?",
        "Giá vé xem phim bao nhiêu?",
        "Kể chuyện cười đi",
        "2 + 2 bằng bao nhiêu",
        "Dạy tôi nấu phở đi",
        "Tỷ giá USD hôm nay là bao nhiêu?",
        "Bạn có thể đặt vé máy bay không?",
        "Tin tức mới nhất hôm nay là gì?",
        "Tôi muốn nghe nhạc",
        "Gọi điện cho tôi lúc 7 giờ sáng",
        "Dịch câu này sang tiếng Anh đi",
        "Bạn tên gì?",
        "Mấy giờ rồi?",
        "Ai là tổng thống Mỹ?",
        "Dạy mình code Python đi",
        "Giúp mình giải bài toán này",
        "Cho mình số điện thoại của rạp CGV",
        "Đặt pizza giúp mình",
        "Bạn có người yêu chưa?",
        "Mình buồn quá, tâm sự với mình đi",
        "Cho mình xem lịch chiếu rạp",
        "Giá popcorn bao nhiêu?",
        "Tải phim ở đâu?",
        "Bạn biết hát không?",
        "Viết thơ tặng bạn gái mình đi",
        "Hôm nay nên ăn gì?",
        "Đường đến rạp phim gần nhất",
        "Bạn có thể chơi game không?",
        "Kể cho mình nghe một câu chuyện",
        "Giúp mình đặt lịch hẹn",
        "Bạn thông minh không?",
        "Sáng nay có gì hot trên mạng?",
        "Mình quên mật khẩu Netflix rồi",
        "Tìm việc làm part-time cho mình",
        "Bạn có biết ngoại ngữ nào không?",
        "Bao giờ thì Tết?",
        "Cho mình hỏi giá Bitcoin",
        "Mình muốn học guitar",
        "Bạn nghĩ gì về cuộc sống?",
    ],
}

_PLACEHOLDER_META = {
    "{ACTOR}"       : ("ACTOR",       "PERSON"),
    "{DIRECTOR}"    : ("DIRECTOR",    "PERSON"),
    "{MOVIE_TITLE}" : ("MOVIE_TITLE", "MOVIE_TITLE"),
    "{GENRE}"       : ("GENRE",       "GENRE"),
    "{YEAR}"        : ("YEAR",        "YEAR"),
}


# Mapping: (placeholder, intent) -> slot_name cho Frame Semantics
# Placeholder khong co trong mapping -> fill nhung KHONG ghi nhan lam argument
_PH_TO_SLOT = {
    "find_movie": {
        "{ACTOR}": "person", "{DIRECTOR}": "person",
        "{MOVIE_TITLE}": "title", "{GENRE}": "genre", "{YEAR}": "year",
    },
    "recommendation": {
    "{ACTOR}": "person", "{MOVIE_TITLE}": "like_movie", "{DIRECTOR}": "person",
    "{GENRE}": "genre", "{YEAR}": "year",
    },
    "movie_info": {
        "{MOVIE_TITLE}": "title",
    },
    "person_info": {
        "{ACTOR}": "name",
    },
    "genre_filter": {
        "{GENRE}": "genre", "{YEAR}": "year",
    },
}

def _empty_entities() -> dict:
    return {label: [] for label in FINAL_ENTITY_LABELS}

_TEENCODE = {
    "không": ["ko", "k", "khong", "hong"],
    "bạn":   ["bn", "b"],
    "phim":  ["film", "fim", "fjm"],
    "tôi":   ["mk", "t", "tui"],
    "muốn":  ["mun", "mún"],
    "xem":   ["coi"],
    "hay":   ["xịn", "đỉnh", "hayy"],
    "gì":    ["j", "z"],
    "được":  ["dc", "đc"],
    "nào":   ["nao"],
}

_TYPO_MAP = {"ph": ["f"], "gi": ["d", "z"], "qu": ["w"], "ch": ["c", "tr"], "tr": ["ch"]}

def _remove_accents(text):
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()

def _apply_teencode(text):
    words = text.split()
    return " ".join(
        random.choice(_TEENCODE[w.lower()])
        if w.lower() in _TEENCODE and random.random() < 0.4 else w
        for w in words
    )

def _apply_typo(text):
    for src, targets in _TYPO_MAP.items():
        if src in text and random.random() < 0.25:
            text = text.replace(src, random.choice(targets), 1)
    return text

def _remove_punctuation(text):
    return re.sub(r"[?!.,]+$", "", text).strip()

def _add_filler(text):
    fillers = ["ơi ", "ê ", "này ", "hey ", "à ", "ừm "]
    if random.random() < 0.3:
        return random.choice(fillers) + text[0].lower() + text[1:]
    return text

def _get_zipfian_choice(items, alpha=1.0):
    """Smoothed Zipfian: alpha=1.0 la Zipf chuan, alpha=0 la uniform."""
    weights = [1.0 / (i + 1) ** alpha for i in range(len(items))]
    return random.choices(items, weights=weights, k=1)[0]

def _get_uniform_choice(items):
    """Uniform random choice."""
    return random.choice(items)

# Reverse map: genre chuẩn -> list alias (dùng cho training augmentation)
_GENRE_SYNONYMS = {}
for _alias, _canonical in GENRE_ALIASES.items():
    _GENRE_SYNONYMS.setdefault(_canonical, []).append(_alias)

# ============================================================
# HÀM GỌI LLM PARAPHRASE (Có Auto-Retry chống Rate Limit)
# ============================================================
def llm_paraphrase(text: str, entities_to_keep: list = None, n_augments: int = 2) -> list:
    if not llm_client:
        return []
    
    constraint = ""
    if entities_to_keep and len(entities_to_keep) > 0:
        constraint = f"BẠN BẮT BUỘC PHẢI GIỮ NGUYÊN TỪNG CHỮ CÁC CỤM TỪ SAU TRONG CÂU TRẢ LỜI: {', '.join(entities_to_keep)}.\n"

    prompt = f"""Viết lại câu sau sang {n_augments} phiên bản khác nhau với văn phong tự nhiên của người Việt Nam (có thể dùng văn phong gen Z, hỏi trống không, hoặc lịch sự).
    Câu gốc: \"{text}\"
    {constraint}
    Trả về kết quả, mỗi câu trên 1 dòng, không đánh số thứ tự, không giải thích gì thêm."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm_client.models.generate_content(
                model='gemma-4-26b-a4b-it', 
                contents=prompt
            )
            
            variations = [line.strip().strip('"').strip("'") for line in response.text.split('\n') if line.strip()]
            
            valid_variations = []
            for var in variations[:n_augments]:
                if entities_to_keep:
                    if all(ent.lower() in var.lower() for ent in entities_to_keep):
                        valid_variations.append(var)
                else:
                    valid_variations.append(var)
                    
            print(f"    [LLM] Nhận được {len(valid_variations)} biến thể hợp lệ từ LLM.")
            print()
            return valid_variations
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"    [!] Quá tải API. Đang đợi 35 giây để thử lại (Lần {attempt+1}/{max_retries})...")
                time.sleep(35)
            else:
                print(f"    [Cảnh báo] Lỗi gọi LLM: {e}")
                break

    return []

def augment_text(text: str, n_augments: int = 3) -> list:
    """Augmentation AN TOAN cho intent data.
    Da loai bo _random_swap va _random_dropout vi chung pha vo ngu phap
    va khien model hoc sai pattern."""
    ops = [
        lambda t: _remove_accents(t),
        lambda t: _apply_teencode(t),
        lambda t: _apply_typo(t),
        lambda t: _remove_punctuation(t),
        lambda t: _add_filler(t),
        lambda t: _apply_teencode(_remove_punctuation(t)),
        lambda t: _remove_accents(_apply_teencode(t)),
        lambda t: _add_filler(_remove_punctuation(t)),
    ]
    seen, augmented = {text}, []
    attempts = 0
    while len(augmented) < n_augments and attempts < n_augments * 5:
        attempts += 1
        result = re.sub(r"\s+", " ", random.choice(ops)(text)).strip()
        if result and result not in seen:
            augmented.append(result)
            seen.add(result)
    return augmented

# ============================================================
# TÍCH HỢP VÀO GENERATE DATASET
# ============================================================

def generate_dataset(
    entities: dict,
    n_samples: int = 150,
    augment: bool = True,
    use_llm: bool = False,
    n_augments_per_sample: int = 2,
) -> list:
    actors    = entities.get("actors",    ["Thành Long", "Lý Liên Kiệt"])
    directors = entities.get("directors", ["Christopher Nolan", "James Cameron"])
    movies    = entities.get("movies",    ["Avengers", "Inception"])
    genres    = entities.get("genres",    ["Hành Động", "Kinh Dị", "Lãng Mạn"])
    years     = [str(y) for y in range(2010, 2025)]

    random.shuffle(actors)
    random.shuffle(directors)
    random.shuffle(movies)
    
    _vocab = {
        "ACTOR": actors, "DIRECTOR": directors,
        "MOVIE_TITLE": movies, "GENRE": genres, "YEAR": years,
    }

    dataset = []
    print(f"Đang sinh dữ liệu Intent. Chế độ LLM: {'BẬT' if use_llm and llm_client else 'TẮT'}")

    for intent, templates in TEMPLATES.items():
        for _ in range(n_samples):
            template = random.choice(templates)
            filled_entities = _empty_entities()
            fmt_kwargs = {k: "" for k in ["ACTOR", "DIRECTOR", "MOVIE_TITLE", "GENRE", "YEAR"]}

            for ph, (key, label) in _PLACEHOLDER_META.items():
                if ph in template:
                    val = _get_zipfian_choice(_vocab[key], alpha=0.5)
                    # 30% lowercase cho tên phim và tên người
                    if label in ("PERSON", "MOVIE_TITLE") and random.random() < 0.3:
                        val = val.lower()
                    fmt_kwargs[key] = val
                    filled_entities[label].append(val)

            text = re.sub(r"\s+", " ", template.format(**fmt_kwargs)).strip()
            
            dataset.append({
                "text"    : text,
                "label"   : intent,
                "label_id": LABEL2ID[intent],
                "entities": filled_entities,
                "source"  : "template",
            })

            if augment:
                aug_texts = []
                if use_llm and llm_client:
                    ents_to_keep = [val for vals in filled_entities.values() for val in vals if val]
                    aug_texts = llm_paraphrase(text, ents_to_keep, n_augments=n_augments_per_sample)
                    if aug_texts:
                        print(aug_texts)
                if not aug_texts:
                    aug_texts = augment_text(text, n_augments=n_augments_per_sample)

                for aug_text in aug_texts:
                    dataset.append({
                        "text"    : aug_text,
                        "label"   : intent,
                        "label_id": LABEL2ID[intent],
                        "entities": filled_entities,
                        "source"  : "llm_augmented" if use_llm and aug_texts else "augmented",
                    })

    random.shuffle(dataset)
    print(f"OK Tổng dataset: {len(dataset)} samples")
    
    _validate_dataset(dataset)
    return dataset


def _validate_dataset(dataset: list):
    """Kiem tra chat luong dataset sau khi sinh."""
    errors = {"empty_text": 0, "invalid_label": 0, "duplicates": 0}
    seen_texts = set()
    for sample in dataset:
        if not sample.get("text", "").strip():
            errors["empty_text"] += 1
        if sample.get("text") in seen_texts:
            errors["duplicates"] += 1
        if sample.get("label") not in INTENT_LABELS:
            errors["invalid_label"] += 1
        seen_texts.add(sample.get("text"))
    
    total = len(dataset)
    print(f"\n[Data Validation] {total} samples:")
    print(f"   Empty text   : {errors['empty_text']}")
    print(f"   Invalid label: {errors['invalid_label']}")
    print(f"   Duplicates   : {errors['duplicates']} ({errors['duplicates']/max(total,1):.1%})")


# Đã thêm dấu Tiếng Việt chuẩn
_NER_TEMPLATES = [
    ("Tìm phim {GENRE} của {ACTOR} ra mắt năm {YEAR}",                     "find_movie"),
    ("{ACTOR} có bộ phim {GENRE} nào chiếu năm {YEAR} không?",             "find_movie"),
    ("Phim {GENRE} nào có {ACTOR} đóng chính năm {YEAR}?",                 "find_movie"),
    ("Cho tôi xem phim {GENRE} do {DIRECTOR} đạo diễn năm {YEAR}",         "find_movie"),
    ("{DIRECTOR} làm phim {GENRE} nào trong năm {YEAR}?",                  "find_movie"),
    ("Tìm phim {MOVIE_TITLE} do {ACTOR} đóng vai chính",                   "find_movie"),
    ("Phim {MOVIE_TITLE} được ra mắt vào năm {YEAR} đúng không?",          "find_movie"),
    ("{ACTOR} và {DIRECTOR} từng cộng tác trong phim {GENRE} nào?",        "find_movie"),
    ("Tìm phim {GENRE} có sự tham gia của {ACTOR} và {DIRECTOR}",          "find_movie"),
    ("Bộ phim {MOVIE_TITLE} thuộc thể loại {GENRE} phải không?",           "find_movie"),
    ("Phim nào của {DIRECTOR} có {ACTOR} đóng vai chính?",                 "find_movie"),
    ("Tìm phim {GENRE} mà {ACTOR} và {DIRECTOR} cùng thực hiện năm {YEAR}","find_movie"),
    ("Có phim {GENRE} nào của {ACTOR} phát hành sau năm {YEAR} không?",    "find_movie"),
    ("Tôi muốn tìm lại phim {GENRE} của {DIRECTOR} hồi năm {YEAR}",        "find_movie"),
    ("{ACTOR} có xuất hiện trong bộ phim {MOVIE_TITLE} không?",            "find_movie"),
    ("Phim {MOVIE_TITLE} do {DIRECTOR} đạo diễn có phải thể loại {GENRE}?","find_movie"),
    ("Tìm các bộ phim {GENRE} do {DIRECTOR} thực hiện từ năm {YEAR}",      "find_movie"),
    ("Phim {GENRE} mới nhất của {ACTOR} là gì?",                           "find_movie"),
    ("Bộ phim nào của {ACTOR} ra đời năm {YEAR} mà thuộc thể loại {GENRE}?","find_movie"),
    ("{DIRECTOR} có làm phim {GENRE} nào cùng {ACTOR} không?",             "find_movie"),
    ("Tìm phim {GENRE} mà {ACTOR} đóng cùng {DIRECTOR} năm {YEAR}",        "find_movie"),
    ("Cho tôi thấy phim {GENRE} của {ACTOR} được chiếu năm {YEAR}",        "find_movie"),
    ("Phim {MOVIE_TITLE} của {DIRECTOR} phát hành năm nào?",               "find_movie"),
    ("Tìm bộ phim {GENRE} do {ACTOR} thủ vai chính vào {YEAR}",            "find_movie"),
    ("{ACTOR} từng đóng phim {GENRE} nào dưới sự chỉ đạo của {DIRECTOR}?", "find_movie"),
    ("Tìm phim có {ACTOR} đóng thuộc thể loại {GENRE}",                    "find_movie"),
    ("Bộ phim {MOVIE_TITLE} gồm những ai trong dàn diễn viên?",            "find_movie"),
    ("Có phim {GENRE} nào của {DIRECTOR} đoạt giải thưởng năm {YEAR} không?","find_movie"),
    ("Tôi đang tìm phim {GENRE} của {ACTOR} hồi còn trẻ",                  "find_movie"),
    ("Phim {GENRE} nào ra đời năm {YEAR} có {DIRECTOR} làm đạo diễn?",     "find_movie"),
    ("{ACTOR} đã đóng bao nhiêu phim {GENRE} rồi?",                        "person_info"),
    ("Phim gần đây nhất của {ACTOR} thuộc thể loại {GENRE} là gì?",        "person_info"),
    ("{ACTOR} đóng phim {GENRE} đầu tiên vào năm {YEAR} là bộ nào?",       "person_info"),
    ("Sự nghiệp của {ACTOR} gắn liền với những bộ phim {GENRE} nào?",      "person_info"),
    ("{ACTOR} có đóng phim cùng {DIRECTOR} bao giờ chưa?",                 "person_info"),
    ("Vai diễn {GENRE} nào của {ACTOR} được đánh giá xuất sắc nhất?",      "person_info"),
    ("{ACTOR} bắt đầu đóng phim {GENRE} từ năm nào?",                      "person_info"),
    ("Liệt kê các phim {GENRE} mà {ACTOR} tham gia từ năm {YEAR} đến nay", "person_info"),
    ("{ACTOR} có phim {GENRE} nào sắp ra mắt không?",                      "person_info"),
    ("Phim {GENRE} nào của {ACTOR} có doanh thu cao nhất?",                "person_info"),
    ("{ACTOR} thường đóng vai gì trong các phim {GENRE}?",                 "person_info"),
    ("Cho tôi biết {ACTOR} đã làm việc với {DIRECTOR} trong những phim nào","person_info"),
    ("{ACTOR} có bao nhiêu phim {GENRE} được phát hành năm {YEAR}?",       "person_info"),
    ("Phim {GENRE} của {ACTOR} và {DIRECTOR} cái nào hay nhất?",           "person_info"),
    ("{ACTOR} đã đóng phim {MOVIE_TITLE} chưa?",                           "person_info"),
    ("Ngoài {MOVIE_TITLE}, {ACTOR} còn phim {GENRE} nào khác không?",      "person_info"),
    ("{ACTOR} đóng phim {GENRE} hay hơn hay phim {MOVIE_TITLE} hay hơn?",  "person_info"),
    ("Kể tên các bộ phim {GENRE} mà {ACTOR} đã đóng trước năm {YEAR}",     "person_info"),
    ("{ACTOR} có được đề cử giải nào cho phim {GENRE} năm {YEAR} không?",  "person_info"),
    ("Vai diễn trong {MOVIE_TITLE} có phải vai nổi tiếng nhất của {ACTOR}?","person_info"),
    ("{ACTOR} có hợp tác với {DIRECTOR} trong phim {GENRE} nào năm {YEAR}?","person_info"),
    ("Phim {GENRE} tiếp theo của {ACTOR} dự kiến ra mắt khi nào?",         "person_info"),
    ("{ACTOR} đóng vai gì trong bộ phim {MOVIE_TITLE}?",                   "person_info"),
    ("Từ năm {YEAR}, {ACTOR} đã xuất hiện trong những phim {GENRE} nào?",  "person_info"),
    ("{ACTOR} nổi bật với thể loại {GENRE} từ năm {YEAR} chưa?",           "person_info"),
    ("Phim {GENRE} của {ACTOR} cái nào được khán giả yêu thích nhất?",     "person_info"),
    ("{ACTOR} có hay đóng phim {GENRE} của đạo diễn {DIRECTOR} không?",    "person_info"),
    ("Tổng hợp sự nghiệp {ACTOR} qua các bộ phim {GENRE} từ năm {YEAR}",   "person_info"),
    ("{ACTOR} đã nhận được giải gì nhờ bộ phim {MOVIE_TITLE}?",            "person_info"),
    ("Đánh giá diễn xuất của {ACTOR} trong phim {GENRE} năm {YEAR}",       "person_info"),
    ("Phim {MOVIE_TITLE} do ai đạo diễn?",                                 "movie_info"),
    ("{MOVIE_TITLE} là phim thuộc thể loại {GENRE} phải không?",           "movie_info"),
    ("Dàn diễn viên trong phim {MOVIE_TITLE} gồm những ai?",               "movie_info"),
    ("Phim {MOVIE_TITLE} có {ACTOR} đóng vai gì?",                         "movie_info"),
    ("Năm {YEAR} bộ phim {MOVIE_TITLE} có được chiếu không?",              "movie_info"),
    ("{MOVIE_TITLE} của đạo diễn {DIRECTOR} kể về chuyện gì?",             "movie_info"),
    ("Điểm Rotten Tomatoes của {MOVIE_TITLE} là bao nhiêu?",               "movie_info"),
    ("Phim {MOVIE_TITLE} có đoạt giải Oscar không?",                       "movie_info"),
    ("{MOVIE_TITLE} được sản xuất bởi hãng phim nào?",                     "movie_info"),
    ("Doanh thu phòng vé của {MOVIE_TITLE} năm {YEAR} là bao nhiêu?",      "movie_info"),
    ("Phim {MOVIE_TITLE} dài bao nhiêu phút?",                             "movie_info"),
    ("{MOVIE_TITLE} có phần 2 chưa?",                                      "movie_info"),
    ("Ngân sách sản xuất phim {MOVIE_TITLE} là bao nhiêu?",                "movie_info"),
    ("Phim {MOVIE_TITLE} có được phân loại phù hợp trẻ em không?",         "movie_info"),
    ("{MOVIE_TITLE} được quay chủ yếu ở đâu?",                             "movie_info"),
    ("Kịch bản phim {MOVIE_TITLE} do {DIRECTOR} viết không?",              "movie_info"),
    ("Phim {MOVIE_TITLE} phát hành tại Việt Nam vào năm {YEAR} chưa?",     "movie_info"),
    ("{MOVIE_TITLE} có bản lồng tiếng Việt không?",                        "movie_info"),
    ("Phim {MOVIE_TITLE} có phải dựa trên tiểu thuyết không?",             "movie_info"),
    ("{MOVIE_TITLE} nhận phản hồi thế nào từ giới phê bình?",              "movie_info"),
    ("Ai viết nhạc phim cho bộ phim {MOVIE_TITLE}?",                       "movie_info"),
    ("Phim {MOVIE_TITLE} có post-credit scene không?",                     "movie_info"),
    ("{MOVIE_TITLE} gây tranh cãi ở điểm nào vậy?",                        "movie_info"),
    ("Hậu trường phim {MOVIE_TITLE} có điều gì thú vị?",                   "movie_info"),
    ("Thông điệp chính của bộ phim {MOVIE_TITLE} là gì?",                  "movie_info"),
    ("{MOVIE_TITLE} có được chiếu tại Cannes năm {YEAR} không?",           "movie_info"),
    ("Phim {MOVIE_TITLE} thuộc series hay phim độc lập?",                  "movie_info"),
    ("{ACTOR} đóng vai phản diện hay chính diện trong {MOVIE_TITLE}?",     "movie_info"),
    ("Phim {MOVIE_TITLE} phù hợp với lứa tuổi nào?",                       "movie_info"),
    ("Cảnh quay nào trong {MOVIE_TITLE} là ấn tượng nhất?",                "movie_info"),
    ("Gợi ý phim {GENRE} giống {MOVIE_TITLE} cho tôi",                     "recommendation"),
    ("Tôi thích {MOVIE_TITLE}, có phim {GENRE} nào tương tự không?",       "recommendation"),
    ("Đề xuất phim {GENRE} hay nhất của {ACTOR} cho tôi xem",              "recommendation"),
    ("Recommend phim {GENRE} của {DIRECTOR} đáng xem nhất",                "recommendation"),
    ("Top phim {GENRE} hay nhất mọi thời đại là gì?",                      "recommendation"),
    ("Gợi ý phim {GENRE} phù hợp xem cuối tuần",                           "recommendation"),
    ("Phim {GENRE} nào của {ACTOR} năm {YEAR} đáng xem nhất?",             "recommendation"),
    ("Tôi muốn xem phim {GENRE} hay, bạn gợi ý {ACTOR} hay {DIRECTOR}?",   "recommendation"),
    ("Phim {GENRE} nào hay hơn {MOVIE_TITLE} không?",                      "recommendation"),
    ("Gợi ý vài bộ phim {GENRE} có nội dung sâu sắc năm {YEAR}",           "recommendation"),
    ("Cho tôi gợi ý phim {GENRE} ít người biết nhưng rất hay",             "recommendation"),
    ("Phim {GENRE} nào của {DIRECTOR} xem được nhất?",                     "recommendation"),
    ("Đề xuất phim {GENRE} kinh điển cho người mới xem thể loại này",      "recommendation"),
    ("Gợi ý phim {GENRE} có twist bất ngờ như {MOVIE_TITLE}",              "recommendation"),
    ("Bạn nghĩ phim {GENRE} nào hay nhất năm {YEAR}?",                     "recommendation"),
    ("Recommend phim {GENRE} cho tôi xem một mình tối nay",                "recommendation"),
    ("Phim {GENRE} nào của {ACTOR} hoặc {DIRECTOR} đang hot nhất?",        "recommendation"),
    ("Gợi ý phim {GENRE} phong cách tương tự {MOVIE_TITLE} ra đời năm {YEAR}","recommendation"),
    ("Có phim {GENRE} nào hay hơn {MOVIE_TITLE} của {DIRECTOR} không?",    "recommendation"),
    ("Đề xuất phim {GENRE} của {ACTOR} phù hợp xem cùng gia đình",         "recommendation"),
    ("Phim {GENRE} mới nhất năm {YEAR} nào đáng bỏ tiền mua vé?",          "recommendation"),
    ("Gợi ý phim {GENRE} có diễn xuất tốt như {ACTOR}",                    "recommendation"),
    ("Cho tôi vài cái tên phim {GENRE} được giới phê bình khen năm {YEAR}","recommendation"),
    ("Tôi chán {MOVIE_TITLE} rồi, gợi ý phim {GENRE} khác đi",             "recommendation"),
    ("Phim {GENRE} nào của {ACTOR} mà khán giả bình thường cũng thích?",   "recommendation"),
    ("Recommend phim {GENRE} dài dưới 2 tiếng, ra mắt năm {YEAR}",         "recommendation"),
    ("Gợi ý phim {GENRE} có nội dung gần giống {MOVIE_TITLE} nhưng mới hơn","recommendation"),
    ("Phim {GENRE} nào của {DIRECTOR} được đánh giá tốt hơn {MOVIE_TITLE}?","recommendation"),
    ("Đề xuất phim {GENRE} có happy ending giống kiểu {MOVIE_TITLE}",      "recommendation"),
    ("Bạn gợi ý gì nếu tôi thích cả {ACTOR} lẫn thể loại {GENRE}?",        "recommendation"),
    ("Danh sách phim {GENRE} hay nhất năm {YEAR}",                         "genre_filter"),
    ("Phim {GENRE} nào đang chiếu rạp năm {YEAR}?",                        "genre_filter"),
    ("Top 10 phim {GENRE} được xem nhiều nhất năm {YEAR}",                 "genre_filter"),
    ("Phim {GENRE} mới ra mắt năm {YEAR} có gì hay?",                      "genre_filter"),
    ("Lọc phim theo thể loại {GENRE} phát hành từ năm {YEAR}",             "genre_filter"),
    ("Cho tôi xem danh sách phim {GENRE} của đạo diễn {DIRECTOR}",         "genre_filter"),
    ("Phim {GENRE} nào của {ACTOR} ra mắt năm {YEAR}?",                    "genre_filter"),
    ("Tìm tất cả phim {GENRE} có {ACTOR} tham gia",                        "genre_filter"),
    ("Phim {GENRE} nào được đánh giá cao nhất trong năm {YEAR}?",          "genre_filter"),
    ("Hiển thị phim {GENRE} do {DIRECTOR} thực hiện từ năm {YEAR}",        "genre_filter"),
    ("Có bao nhiêu phim {GENRE} ra mắt năm {YEAR}?",                       "genre_filter"),
    ("Phim {GENRE} nào của {DIRECTOR} được khán giả yêu thích nhất?",      "genre_filter"),
    ("Cho tôi thấy phim {GENRE} có điểm IMDb trên 8 năm {YEAR}",           "genre_filter"),
    ("Danh sách phim {GENRE} do {ACTOR} đóng chính từ năm {YEAR}",         "genre_filter"),
    ("Phim {GENRE} nào đoạt giải thưởng lớn trong năm {YEAR}?",            "genre_filter"),
    ("Tổng hợp phim {GENRE} của {DIRECTOR} trong thập kỷ qua",             "genre_filter"),
    ("Phim {GENRE} nào của {ACTOR} có doanh thu cao nhất năm {YEAR}?",     "genre_filter"),
    ("Lọc phim {GENRE} phù hợp cho khán giả trên 18 tuổi năm {YEAR}",      "genre_filter"),
    ("Cho tôi danh sách phim {GENRE} sản xuất ngoài Hollywood năm {YEAR}", "genre_filter"),
    ("Phim {GENRE} nào của {DIRECTOR} và {ACTOR} cùng thực hiện?",         "genre_filter"),
    ("Tìm phim {GENRE} có rating Rotten Tomatoes cao nhất năm {YEAR}",     "genre_filter"),
    ("Danh sách phim {GENRE} kinh điển trước năm {YEAR}",                  "genre_filter"),
    ("Có phim {GENRE} nào của {ACTOR} chưa được chiếu tại Việt Nam không?","genre_filter"),
    ("Phim {GENRE} nào được khán giả Việt Nam yêu thích nhất năm {YEAR}?", "genre_filter"),
    ("Lọc phim {GENRE} theo đạo diễn {DIRECTOR} có rating cao",            "genre_filter"),
    ("Danh sách phim {GENRE} có thời lượng dưới 90 phút ra mắt năm {YEAR}","genre_filter"),
    ("Phim {GENRE} nào của {ACTOR} chưa có phần tiếp theo?",               "genre_filter"),
    ("Tổng hợp phim {GENRE} đoạt giải Oscar từ năm {YEAR} đến nay",        "genre_filter"),
    ("Cho tôi xem phim {GENRE} nào được {DIRECTOR} làm lại từ phim cũ",    "genre_filter"),
    ("Phim {GENRE} nào do cả {ACTOR} và {DIRECTOR} thực hiện năm {YEAR}?", "genre_filter"),
    ("Tìm giúp tôi bộ phim {MOVIE_TITLE}",                                 "find_movie"),
    ("Có phim nào tên {MOVIE_TITLE} không?",                               "find_movie"),
    ("{ACTOR} đóng phim gì năm {YEAR}?",                                   "find_movie"),
    ("Phim {GENRE} có {ACTOR} đóng hồi năm {YEAR} tên gì?",                "find_movie"),
    ("Tìm phim {MOVIE_TITLE} cho tôi xem với",                             "find_movie"),
    ("Cho tôi biết phim {GENRE} nào có {DIRECTOR} đạo diễn và {ACTOR} đóng", "find_movie"),
    ("{DIRECTOR} làm đạo diễn phim nào có {ACTOR} đóng chính năm {YEAR}?", "find_movie"),
    ("Phim {GENRE} do {ACTOR} đóng cùng đạo diễn {DIRECTOR} tên gì nhỉ?",  "find_movie"),
    ("Mình nhớ có phim {GENRE} tên {MOVIE_TITLE}, tìm giúp mình",          "find_movie"),
    ("Năm {YEAR} có phim {GENRE} nào của {DIRECTOR} không nhỉ?",           "find_movie"),
    ("{ACTOR} năm nay đóng phim gì?",                                      "person_info"),
    ("Tiểu sử của {ACTOR} như thế nào?",                                   "person_info"),
    ("{ACTOR} có hay đóng phim {GENRE} không?",                            "person_info"),
    ("Cho tôi biết về {ACTOR}",                                            "person_info"),
    ("{ACTOR} đóng cặp với ai trong phim {MOVIE_TITLE}?",                  "person_info"),
    ("Phim nào giúp {ACTOR} nổi tiếng nhất?",                              "person_info"),
    ("{ACTOR} và {DIRECTOR} đã hợp tác bao nhiêu lần?",                    "person_info"),
    ("Sắp tới {ACTOR} có phim {GENRE} nào mới không?",                     "person_info"),
    ("{ACTOR} có giỏi đóng phim {GENRE} không?",                           "person_info"),
    ("Tổng hợp phim của {ACTOR} từ năm {YEAR} đến giờ",                    "person_info"),
    ("Nội dung {MOVIE_TITLE} nói về gì?",                                  "movie_info"),
    ("{MOVIE_TITLE} có {ACTOR} đóng vai gì?",                              "movie_info"),
    ("Cho tôi thông tin chi tiết về {MOVIE_TITLE}",                        "movie_info"),
    ("{MOVIE_TITLE} thuộc thể loại {GENRE} à?",                            "movie_info"),
    ("Đạo diễn {DIRECTOR} làm {MOVIE_TITLE} năm bao nhiêu?",               "movie_info"),
    ("Rating của {MOVIE_TITLE} trên TMDB là mấy?",                         "movie_info"),
    ("{MOVIE_TITLE} có bao nhiêu phần rồi vậy?",                           "movie_info"),
    ("Phim {MOVIE_TITLE} chiếu năm {YEAR} đúng không?",                    "movie_info"),
    ("{MOVIE_TITLE} được quay ở nước nào?",                                "movie_info"),
    ("Ai là nhà sản xuất của {MOVIE_TITLE}?",                              "movie_info"),
    ("Gợi ý phim {GENRE} hay đi bạn",                                      "recommendation"),
    ("Phim nào giống {MOVIE_TITLE} vậy?",                                  "recommendation"),
    ("{ACTOR} có phim {GENRE} nào đáng xem không?",                        "recommendation"),
    ("Cho tôi vài phim {GENRE} hay của {DIRECTOR}",                        "recommendation"),
    ("Suggest phim {GENRE} năm {YEAR} đi bạn",                             "recommendation"),
    ("Tôi thích phim của {ACTOR}, gợi ý thêm đi",                          "recommendation"),
    ("Phim {GENRE} nào hay như {MOVIE_TITLE}?",                            "recommendation"),
    ("Recommend phim {GENRE} do {DIRECTOR} làm đi",                        "recommendation"),
    ("Hôm nay rảnh, gợi ý phim {GENRE} xem với",                           "recommendation"),
    ("Phim nào hay nhất năm {YEAR} thể loại {GENRE}?",                     "recommendation"),
    ("Cho xem phim {GENRE} đi",                                            "genre_filter"),
    ("Liệt kê phim {GENRE} cho tôi",                                       "genre_filter"),
    ("Phim {GENRE} năm {YEAR} có gì hay?",                                 "genre_filter"),
    ("Tìm phim {GENRE} có {ACTOR} đóng",                                   "genre_filter"),
    ("Phim {GENRE} nào của {DIRECTOR} hay nhất?",                          "genre_filter"),
    ("Top phim {GENRE} đáng xem năm {YEAR}",                               "genre_filter"),
    ("Cho tôi xem tất cả phim {GENRE} của {DIRECTOR}",                     "genre_filter"),
    ("Phim {GENRE} nào đang hot năm {YEAR}?",                              "genre_filter"),
    ("Danh sách phim {GENRE} có {ACTOR} tham gia năm {YEAR}",              "genre_filter"),
    ("Lọc phim {GENRE} ra mắt từ năm {YEAR} trở đi",                       "genre_filter"),
    ("Mình muốn tìm phim {GENRE} có {ACTOR} đóng gần năm {YEAR}",          "find_movie"),
    ("Kiếm giúp tôi bộ {GENRE} của {DIRECTOR} ra rạp năm {YEAR}",          "find_movie"),
    ("Phim {MOVIE_TITLE} có phải do {DIRECTOR} cầm trịch không?",          "find_movie"),
    ("Có bộ nào {ACTOR} tham gia mà thuộc dòng {GENRE} không?",            "find_movie"),
    ("Tìm phim của {ACTOR} phát hành vào {YEAR}",                          "find_movie"),
    ("Tôi cần phim {GENRE} do {DIRECTOR} làm để xem tối nay",              "find_movie"),
    ("Bộ {MOVIE_TITLE} thuộc dòng {GENRE} hay thể loại khác?",             "find_movie"),
    ("{DIRECTOR} có phim nào với {ACTOR} trong năm {YEAR}?",               "find_movie"),
    ("Tìm giùm phim {GENRE} có bối cảnh năm {YEAR} của {DIRECTOR}",        "find_movie"),
    ("Kiếm phim {MOVIE_TITLE} xem có {ACTOR} xuất hiện không",             "find_movie"),
    ("Có phim {GENRE} nào của {ACTOR} hợp tác cùng {DIRECTOR} không?",     "find_movie"),
    ("Tìm tên phim {GENRE} mà {ACTOR} đóng hồi {YEAR}",                    "find_movie"),
    ("{ACTOR} xuất hiện trong phim nào của {DIRECTOR}?",                   "find_movie"),
    ("Mình đang kiếm phim {MOVIE_TITLE} ra mắt năm {YEAR}",                "find_movie"),
    ("Cho tôi bộ phim {GENRE} nổi bật của {DIRECTOR}",                     "find_movie"),
    ("Tìm phim có {ACTOR} đóng thuộc thể loại {GENRE}",                    "find_movie"),
    ("Có bộ {MOVIE_TITLE} nào do {DIRECTOR} đạo diễn năm {YEAR} không?",   "find_movie"),
    ("Phim của {ACTOR} năm {YEAR} mà thuộc dòng {GENRE} là gì?",           "find_movie"),
    ("Tìm tác phẩm {GENRE} của {DIRECTOR} có {ACTOR} tham gia",            "find_movie"),
    ("Bộ phim nào mang tên {MOVIE_TITLE} và do {ACTOR} đóng?",             "find_movie"),
    ("Tra cứu phim {GENRE} chiếu khoảng năm {YEAR} của {ACTOR}",           "find_movie"),
    ("Có phim nào tên {MOVIE_TITLE} thuộc thể loại {GENRE} không?",        "find_movie"),
    ("Tìm phim {GENRE} mà {DIRECTOR} làm chung với {ACTOR}",               "find_movie"),
    ("Phim nào năm {YEAR} của {DIRECTOR} có {ACTOR} đóng?",                "find_movie"),
    ("Mình muốn coi lại phim {MOVIE_TITLE} của {DIRECTOR}",                "find_movie"),
    ("Kiếm phim {GENRE} hợp gu có {ACTOR} xuất hiện",                      "find_movie"),
    ("Phim {GENRE} nào do {DIRECTOR} chỉ đạo và lên sóng năm {YEAR}?",     "find_movie"),
    ("Tìm giúp phim của {ACTOR} tên gần giống {MOVIE_TITLE}",              "find_movie"),
    ("Có bộ phim {GENRE} nào của {ACTOR} ra mắt trước {YEAR} không?",      "find_movie"),
    ("Phim {MOVIE_TITLE} với {ACTOR} là cùng một bộ phải không?",          "find_movie"),
    ("Tôi đang tìm bộ {GENRE} mà {DIRECTOR} ra mắt năm {YEAR}",            "find_movie"),
    ("Bộ phim nào của {ACTOR} thuộc dạng {GENRE} đáng nhớ nhất?",          "find_movie"),
    ("Tìm phim {MOVIE_TITLE} xem có phải phim {GENRE} không",              "find_movie"),
    ("Có phim nào do {DIRECTOR} làm mà {ACTOR} đóng và ra năm {YEAR}?",    "find_movie"),
    ("Tìm bộ phim của {ACTOR} hợp tác với {DIRECTOR}",                     "find_movie"),
    ("Tôi nhớ một phim {GENRE} có {ACTOR}, tìm giúp với",                  "find_movie"),
    ("Kiếm tác phẩm của {DIRECTOR} thuộc thể loại {GENRE}",                "find_movie"),
    ("Phim {MOVIE_TITLE} thuộc năm {YEAR} có đúng không?",                 "find_movie"),
    ("Có bộ phim năm {YEAR} nào của {ACTOR} tên {MOVIE_TITLE} không?",     "find_movie"),
    ("Tìm phim {GENRE} năm {YEAR} có đạo diễn là {DIRECTOR}",              "find_movie"),
    ("Mình cần phim có {ACTOR} đóng chính và tên {MOVIE_TITLE}",           "find_movie"),
    ("Bộ {MOVIE_TITLE} do {DIRECTOR} làm có phải ra năm {YEAR} không?",    "find_movie"),
    ("Có phim {GENRE} nào năm {YEAR} mà {ACTOR} dẫn dắt không?",           "find_movie"),
    ("Tìm phim của {DIRECTOR} có tên {MOVIE_TITLE}",                       "find_movie"),
    ("Phim nào của {ACTOR} thuộc nhóm {GENRE} và khá mới?",                "find_movie"),
    ("Tìm phim {GENRE} phát hành quanh {YEAR} với {ACTOR}",                "find_movie"),
    ("Có phim nào của {DIRECTOR} trùng với tên {MOVIE_TITLE} không?",      "find_movie"),
    ("Tra giúp bộ {MOVIE_TITLE} xem thuộc thể loại {GENRE}",               "find_movie"),
    ("Kiếm phim {GENRE} do {DIRECTOR} làm cùng dàn cast có {ACTOR}",       "find_movie"),
    ("Tôi muốn xem phim {ACTOR} đóng mà ra năm {YEAR}",                    "find_movie"),
    ("Tìm một bộ {GENRE} có cả {ACTOR} lẫn {DIRECTOR}",                    "find_movie"),
    ("Bộ phim {MOVIE_TITLE} có nằm trong danh sách phim {GENRE} không?",   "find_movie"),
    ("Có phim nào của {ACTOR} tên {MOVIE_TITLE} phát hành {YEAR} không?",  "find_movie"),
    ("Tìm phim {DIRECTOR} đạo diễn cho thể loại {GENRE}",                  "find_movie"),
    ("Phim {GENRE} nào của {ACTOR} ra mắt sau {YEAR}?",                    "find_movie"),
    ("Mình đang kiếm bộ phim tên {MOVIE_TITLE} của {ACTOR}",               "find_movie"),
    ("Có bộ {GENRE} nào của {DIRECTOR} mà tôi nên tìm trước {YEAR}?",      "find_movie"),
    ("Tìm phim {ACTOR} tham gia dưới trướng {DIRECTOR}",                   "find_movie"),
    ("Kiểm tra xem {MOVIE_TITLE} có phải phim của {DIRECTOR} không",       "find_movie"),
    ("Tìm phim {GENRE} có {ACTOR} xuất hiện chung với ê-kíp của {DIRECTOR}", "find_movie"),
    ("Diễn viên {ACTOR} nổi bật ở dòng phim {GENRE} nào?",                 "person_info"),
    ("{ACTOR} có phim mới ra khoảng năm {YEAR} không?",                    "person_info"),
    ("Vai diễn nào của {ACTOR} trong {MOVIE_TITLE} được nhớ nhất?",        "person_info"),
    ("{ACTOR} từng làm việc với {DIRECTOR} ở dự án nào?",                  "person_info"),
    ("Tôi muốn biết {ACTOR} hợp với thể loại {GENRE} ra sao",              "person_info"),
    ("{ACTOR} có bao nhiêu phim công chiếu năm {YEAR}?",                   "person_info"),
    ("Bộ {MOVIE_TITLE} có phải phim hay nhất của {ACTOR} không?",          "person_info"),
    ("Sự nghiệp {ACTOR} gắn với đạo diễn {DIRECTOR} thế nào?",             "person_info"),
    ("{ACTOR} từng thử sức ở phim {GENRE} từ năm {YEAR} chưa?",            "person_info"),
    ("Cho tôi profile nhanh của {ACTOR}",                                  "person_info"),
    ("{ACTOR} nổi lên sau bộ {MOVIE_TITLE} đúng không?",                   "person_info"),
    ("Từ {YEAR} đến nay {ACTOR} đóng những phim gì?",                      "person_info"),
    ("{ACTOR} với {DIRECTOR} có phải cộng sự quen thuộc không?",           "person_info"),
    ("Diễn xuất của {ACTOR} trong mảng {GENRE} có tốt không?",             "person_info"),
    ("{ACTOR} đã từng nhận giải nhờ {MOVIE_TITLE} chưa?",                  "person_info"),
    ("Có nên bắt đầu xem {ACTOR} qua phim {GENRE} nào?",                   "person_info"),
    ("{ACTOR} có đóng phim do {DIRECTOR} sản xuất không?",                 "person_info"),
    ("Gần đây {ACTOR} có quay lại với thể loại {GENRE} không?",            "person_info"),
    ("Tên tuổi {ACTOR} gắn với phim nào nhất?",                            "person_info"),
    ("{ACTOR} tham gia dự án nào vào năm {YEAR}?",                         "person_info"),
    ("Phim {MOVIE_TITLE} có giúp {ACTOR} đổi hình tượng không?",           "person_info"),
    ("{ACTOR} và {DIRECTOR} từng hợp tác bao nhiêu phim?",                 "person_info"),
    ("{ACTOR} có hợp với vai trong phim {GENRE} không?",                   "person_info"),
    ("Sau năm {YEAR} {ACTOR} có nổi hơn không?",                           "person_info"),
    ("{ACTOR} từng đóng chính trong bộ {MOVIE_TITLE} à?",                  "person_info"),
    ("Tôi cần danh sách phim đáng chú ý của {ACTOR}",                      "person_info"),
    ("{ACTOR} có thiên hướng chọn phim {GENRE} hay không?",                "person_info"),
    ("{ACTOR} xuất hiện trong phim nào của {DIRECTOR} nhiều người biết?",  "person_info"),
    ("Giai đoạn năm {YEAR} của {ACTOR} có gì nổi bật?",                    "person_info"),
    ("{ACTOR} đóng phim nào mà khán giả nhớ mãi?",                         "person_info"),
    ("Với người mới thì nên xem phim nào của {ACTOR}?",                    "person_info"),
    ("{ACTOR} có thử đóng phim {GENRE} pha hài chưa?",                     "person_info"),
    ("Từ phim {MOVIE_TITLE} có thể đánh giá gì về {ACTOR}?",               "person_info"),
    ("{ACTOR} với {DIRECTOR} ai là người hợp tác quan trọng hơn?",         "person_info"),
    ("{ACTOR} có vai diễn đỉnh cao nào trong năm {YEAR}?",                 "person_info"),
    ("Cho tôi biết hành trình nghề nghiệp của {ACTOR}",                    "person_info"),
    ("{ACTOR} có từng tham gia thương hiệu phim {MOVIE_TITLE} không?",     "person_info"),
    ("Phim {GENRE} nào giúp {ACTOR} được công nhận?",                      "person_info"),
    ("{ACTOR} và {DIRECTOR} có gu làm phim giống nhau không?",             "person_info"),
    ("Năm {YEAR} {ACTOR} đóng vai gì đáng nhớ nhất?",                      "person_info"),
    ("{ACTOR} có phim nào vượt mốc doanh thu lớn không?",                  "person_info"),
    ("Bộ {MOVIE_TITLE} xếp hạng thế nào trong sự nghiệp {ACTOR}?",         "person_info"),
    ("{ACTOR} từng chuyển từ phim {GENRE} sang phim khác ra sao?",         "person_info"),
    ("Thành tựu lớn nhất của {ACTOR} là gì?",                              "person_info"),
    ("{ACTOR} có thường đóng phim cùng một kiểu đạo diễn như {DIRECTOR} không?", "person_info"),
    ("Từ sau {YEAR} phong cách của {ACTOR} thay đổi thế nào?",             "person_info"),
    ("{ACTOR} có tác phẩm nào cùng tông với {MOVIE_TITLE} không?",         "person_info"),
    ("Mảng phim {GENRE} có phải sở trường của {ACTOR}?",                   "person_info"),
    ("{ACTOR} có vai phản diện nào nổi bật không?",                        "person_info"),
    ("Những phim nào của {ACTOR} nên xem trước tiên?",                     "person_info"),
    ("{ACTOR} từng gây bất ngờ trong dự án {GENRE} nào?",                  "person_info"),
    ("Đạo diễn {DIRECTOR} đánh giá {ACTOR} ra sao qua các phim?",          "person_info"),
    ("{ACTOR} có phim đáng nhớ nào phát hành năm {YEAR}?",                 "person_info"),
    ("Vai trong {MOVIE_TITLE} có phải bước ngoặt của {ACTOR}?",            "person_info"),
    ("{ACTOR} hợp đóng phim thương mại hay nghệ thuật hơn?",               "person_info"),
    ("{ACTOR} có dự án nào cùng {DIRECTOR} sắp công bố không?",            "person_info"),
    ("Khán giả thích {ACTOR} nhất ở phim {GENRE} nào?",                    "person_info"),
    ("{ACTOR} có giữ phong độ từ năm {YEAR} đến nay không?",               "person_info"),
    ("Tên {ACTOR} thường gắn với bộ phim nào?",                            "person_info"),
    ("Nếu thích {ACTOR} thì nên xem phim nào đầu tiên?",                   "person_info"),
    ("Tóm tắt nhanh phim {MOVIE_TITLE} cho tôi",                           "movie_info"),
    ("{MOVIE_TITLE} do {DIRECTOR} làm có đáng chú ý không?",               "movie_info"),
    ("{MOVIE_TITLE} thuộc nhánh {GENRE} nào rõ nhất?",                     "movie_info"),
    ("{ACTOR} trong {MOVIE_TITLE} đóng nhân vật nào?",                     "movie_info"),
    ("{MOVIE_TITLE} phát hành chính thức năm {YEAR} à?",                   "movie_info"),
    ("Tôi muốn biết điểm mạnh của phim {MOVIE_TITLE}",                     "movie_info"),
    ("Bộ {MOVIE_TITLE} có gì nổi bật so với phim cùng thể loại {GENRE}?",  "movie_info"),
    ("{MOVIE_TITLE} của {DIRECTOR} có tiết tấu nhanh không?",              "movie_info"),
    ("{MOVIE_TITLE} có phù hợp cho người thích {GENRE} không?",            "movie_info"),
    ("Xem {MOVIE_TITLE} thì nên kỳ vọng điều gì?",                         "movie_info"),
    ("{MOVIE_TITLE} có phải phim ăn khách của {ACTOR} không?",             "movie_info"),
    ("Phim {MOVIE_TITLE} kéo dài tầm bao lâu?",                            "movie_info"),
    ("{DIRECTOR} gửi gắm thông điệp gì trong {MOVIE_TITLE}?",              "movie_info"),
    ("{MOVIE_TITLE} có nhiều cảnh hành động không?",                       "movie_info"),
    ("Không khí của {MOVIE_TITLE} có đậm chất {GENRE} không?",             "movie_info"),
    ("{MOVIE_TITLE} có phần hình ảnh đáng khen không?",                    "movie_info"),
    ("{ACTOR} thể hiện tốt không trong {MOVIE_TITLE}?",                    "movie_info"),
    ("{MOVIE_TITLE} từng gây sốt vào năm {YEAR} phải không?",              "movie_info"),
    ("Chất lượng kịch bản của {MOVIE_TITLE} ra sao?",                      "movie_info"),
    ("{MOVIE_TITLE} có dựa vào nguyên tác nào không?",                     "movie_info"),
    ("Phim {MOVIE_TITLE} được khán giả nhận xét thế nào?",                 "movie_info"),
    ("{MOVIE_TITLE} có phải tác phẩm tiêu biểu của {DIRECTOR}?",           "movie_info"),
    ("Trong {MOVIE_TITLE}, {ACTOR} có nhiều đất diễn không?",              "movie_info"),
    ("{MOVIE_TITLE} có hợp xem cuối tuần không?",                          "movie_info"),
    ("Cốt lõi câu chuyện của {MOVIE_TITLE} là gì?",                        "movie_info"),
    ("{MOVIE_TITLE} có mốc doanh thu lớn trong năm {YEAR} không?",         "movie_info"),
    ("Phim {MOVIE_TITLE} có nhiều plot twist không?",                      "movie_info"),
    ("Âm nhạc của {MOVIE_TITLE} có ấn tượng không?",                       "movie_info"),
    ("{MOVIE_TITLE} khác gì so với các phim {GENRE} khác?",                "movie_info"),
    ("Tông màu của {MOVIE_TITLE} có u tối không?",                         "movie_info"),
    ("{MOVIE_TITLE} có đáng xem vì {ACTOR} không?",                        "movie_info"),
    ("{MOVIE_TITLE} là phim độc lập hay thuộc franchise?",                 "movie_info"),
    ("Người xem thường nhớ gì nhất ở {MOVIE_TITLE}?",                      "movie_info"),
    ("{MOVIE_TITLE} có phù hợp để xem cùng gia đình không?",               "movie_info"),
    ("Diễn biến của {MOVIE_TITLE} có dễ theo dõi không?",                  "movie_info"),
    ("{MOVIE_TITLE} có phải phim thành công của năm {YEAR}?",              "movie_info"),
    ("{MOVIE_TITLE} mang phong cách quen thuộc của {DIRECTOR} chứ?",       "movie_info"),
    ("Điều gì khiến {MOVIE_TITLE} nổi bật trong dòng {GENRE}?",            "movie_info"),
    ("{MOVIE_TITLE} có hậu trường nào thú vị không?",                      "movie_info"),
    ("{ACTOR} có tạo điểm nhấn lớn trong phim {MOVIE_TITLE} không?",       "movie_info"),
    ("Xếp hạng của {MOVIE_TITLE} trong sự nghiệp {DIRECTOR} thế nào?",     "movie_info"),
    ("{MOVIE_TITLE} có đoạn kết dễ hiểu hay mở?",                          "movie_info"),
    ("Phim {MOVIE_TITLE} xem một lần có đủ hiểu không?",                   "movie_info"),
    ("{MOVIE_TITLE} có hợp với người mới xem thể loại {GENRE}?",           "movie_info"),
    ("{MOVIE_TITLE} có nhịp phim nhanh hay chậm?",                         "movie_info"),
    ("{MOVIE_TITLE} ra rạp năm {YEAR} hay phát hành nền tảng số?",         "movie_info"),
    ("{MOVIE_TITLE} có đáng xem vì phần hình ảnh không?",                  "movie_info"),
    ("{DIRECTOR} có tự đổi phong cách ở {MOVIE_TITLE} không?",             "movie_info"),
    ("{MOVIE_TITLE} có tuyến nhân vật của {ACTOR} hấp dẫn chứ?",           "movie_info"),
    ("Chất lượng tổng thể của {MOVIE_TITLE} được đánh giá ra sao?",        "movie_info"),
    ("{MOVIE_TITLE} có yếu tố cảm xúc mạnh không?",                        "movie_info"),
    ("Phim {MOVIE_TITLE} có cảnh nào mang tính biểu tượng?",               "movie_info"),
    ("{MOVIE_TITLE} có hợp với khán giả thích phim {GENRE} cổ điển?",      "movie_info"),
    ("{MOVIE_TITLE} có làm tốt phần thế giới quan không?",                 "movie_info"),
    ("{ACTOR} trong {MOVIE_TITLE} có phá cách không?",                     "movie_info"),
    ("{MOVIE_TITLE} có phải phim dễ recommend không?",                     "movie_info"),
    ("Đâu là lý do nên xem {MOVIE_TITLE}?",                                "movie_info"),
    ("{MOVIE_TITLE} có bị chê ở điểm nào đáng kể?",                        "movie_info"),
    ("{DIRECTOR} đầu tư gì mạnh tay cho {MOVIE_TITLE}?",                   "movie_info"),
    ("{MOVIE_TITLE} có giữ được sức hút từ năm {YEAR} đến giờ không?",     "movie_info"),
    ("Gợi ý cho tôi vài phim {GENRE} dễ xem",                              "recommendation"),
    ("Nếu thích {ACTOR} thì nên xem phim nào tiếp?",                       "recommendation"),
    ("Tôi mê {MOVIE_TITLE}, recommend thêm vài phim tương tự",             "recommendation"),
    ("Cho tôi phim của {DIRECTOR} mà ai cũng khen",                        "recommendation"),
    ("Tìm phim {GENRE} hợp để xem tối nay",                                "recommendation"),
    ("Có phim nào giống {MOVIE_TITLE} nhưng mới hơn không?",               "recommendation"),
    ("Gợi ý phim {GENRE} dành cho người mới bắt đầu",                      "recommendation"),
    ("Nếu thích {ACTOR} và {GENRE} thì nên xem gì?",                       "recommendation"),
    ("Recommend phim của {DIRECTOR} có nhịp nhanh",                        "recommendation"),
    ("Tôi cần vài phim hay ra mắt năm {YEAR}",                             "recommendation"),
    ("Đề cử phim {GENRE} có cảm xúc mạnh",                                 "recommendation"),
    ("Gợi ý bộ phim đáng xem nhất của {ACTOR}",                            "recommendation"),
    ("Có phim nào cùng vibe với {MOVIE_TITLE} không?",                     "recommendation"),
    ("Muốn xem phim của {DIRECTOR} thì nên bắt đầu từ đâu?",               "recommendation"),
    ("Recommend phim {GENRE} có rating ổn năm {YEAR}",                     "recommendation"),
    ("Cho tôi một phim dễ nghiện kiểu {MOVIE_TITLE}",                      "recommendation"),
    ("Gợi ý phim {GENRE} ít bị overrated",                                 "recommendation"),
    ("Tôi muốn xem phim của {ACTOR} mà không bị nặng não",                 "recommendation"),
    ("Phim nào của {DIRECTOR} hợp xem cuối tuần?",                         "recommendation"),
    ("Đề xuất phim {GENRE} vừa hay vừa dễ hiểu",                           "recommendation"),
    ("Có phim nào của {ACTOR} hay hơn {MOVIE_TITLE} không?",               "recommendation"),
    ("Recommend phim phát hành quanh năm {YEAR}",                          "recommendation"),
    ("Gợi ý phim {GENRE} có phần hình ảnh đẹp",                            "recommendation"),
    ("Tôi đang cần phim của {DIRECTOR} để cày",                            "recommendation"),
    ("Cho tôi vài phim tương tự {MOVIE_TITLE} nhưng vui hơn",              "recommendation"),
    ("Phim {GENRE} nào đáng xem chung với bạn bè?",                        "recommendation"),
    ("Đề xuất phim {ACTOR} đóng mà ít người biết",                         "recommendation"),
    ("Gợi ý tác phẩm tốt nhất của {DIRECTOR}",                             "recommendation"),
    ("Tôi thích phim nhịp nhanh, có bộ {GENRE} nào không?",                "recommendation"),
    ("Recommend phim {GENRE} có ending đã",                                "recommendation"),
    ("Cho tôi phim nào của {ACTOR} dễ xem nhất",                           "recommendation"),
    ("Gợi ý phim có màu sắc giống {MOVIE_TITLE}",                          "recommendation"),
    ("Có phim {GENRE} nào hay ra trong năm {YEAR} không?",                 "recommendation"),
    ("Đề xuất bộ phim {DIRECTOR} làm mà ít bị chê",                        "recommendation"),
    ("Nếu chán {MOVIE_TITLE} thì chuyển sang phim nào?",                   "recommendation"),
    ("Recommend phim {GENRE} có diễn xuất nổi bật",                        "recommendation"),
    ("Gợi ý phim của {ACTOR} xem để giải trí",                             "recommendation"),
    ("Tôi muốn phim của {DIRECTOR} có kịch bản chắc tay",                  "recommendation"),
    ("Phim nào hợp với người thích {GENRE} nhẹ đô?",                       "recommendation"),
    ("Cho tôi top phim đáng xem năm {YEAR}",                               "recommendation"),
    ("Gợi ý phim tương tự {MOVIE_TITLE} nhưng căng hơn",                   "recommendation"),
    ("Recommend phim {GENRE} không quá dài",                               "recommendation"),
    ("Có phim của {ACTOR} nào xem xong nhớ lâu không?",                    "recommendation"),
    ("Gợi ý phim {DIRECTOR} làm có nhiều cảm xúc",                         "recommendation"),
    ("Tôi cần phim {GENRE} hợp xem lúc khuya",                             "recommendation"),
    ("Đề xuất phim {GENRE} được khen nhiều năm {YEAR}",                    "recommendation"),
    ("Cho tôi một phim kiểu {MOVIE_TITLE} nhưng lạ hơn",                   "recommendation"),
    ("Recommend phim của {ACTOR} có doanh thu cao",                        "recommendation"),
    ("Có bộ nào của {DIRECTOR} vừa nghệ thuật vừa dễ xem không?",          "recommendation"),
    ("Gợi ý phim {GENRE} có dàn cast tốt",                                 "recommendation"),
    ("Tôi muốn vài phim đáng tiền nhất năm {YEAR}",                        "recommendation"),
    ("Phim nào của {ACTOR} hợp giới thiệu cho người mới?",                 "recommendation"),
    ("Recommend phim như {MOVIE_TITLE} nhưng đỡ u tối hơn",                "recommendation"),
    ("Cho tôi phim {GENRE} có tiết tấu cuốn",                              "recommendation"),
    ("Gợi ý phim do {DIRECTOR} làm mà fan hay nhắc",                       "recommendation"),
    ("Tôi cần phim {ACTOR} đóng để xem nhanh trong tối nay",               "recommendation"),
    ("Recommend phim {GENRE} nhiều người underrated",                      "recommendation"),
    ("Bộ phim nào năm {YEAR} nên ưu tiên xem trước?",                      "recommendation"),
    ("Nếu thích gu của {DIRECTOR} thì có phim nào bắt buộc xem?",          "recommendation"),
    ("Gợi ý phim {GENRE} khiến người xem dễ nghiện",                       "recommendation"),
    ("Lọc cho tôi phim {GENRE} mới nhất",                                  "genre_filter"),
    ("Cho xem các phim {GENRE} nổi nhất năm {YEAR}",                      "genre_filter"),
    ("Tìm phim {GENRE} có {ACTOR} tham gia gần đây",                       "genre_filter"),
    ("Lọc phim của {DIRECTOR} theo thể loại {GENRE}",                     "genre_filter"),
    ("Có bao nhiêu phim {GENRE} ra năm {YEAR}?",                           "genre_filter"),
    ("Hiển thị các phim {GENRE} dễ xem nhất",                              "genre_filter"),
    ("Danh sách phim {GENRE} có doanh thu cao",                            "genre_filter"),
    ("Tìm phim {GENRE} có tên gần giống {MOVIE_TITLE}",                    "genre_filter"),
    ("Cho tôi phim {GENRE} do {ACTOR} đóng chính",                         "genre_filter"),
    ("Lọc phim {GENRE} ra mắt sau năm {YEAR}",                             "genre_filter"),
    ("Hiển thị phim {GENRE} của {DIRECTOR} trước {YEAR}",                  "genre_filter"),
    ("Tôi cần danh sách phim {GENRE} hot hiện tại",                        "genre_filter"),
    ("Tìm các phim {GENRE} nhiều người xem",                               "genre_filter"),
    ("Cho tôi bộ lọc phim {GENRE} theo năm {YEAR}",                        "genre_filter"),
    ("Có phim {GENRE} nào của {ACTOR} đang nổi không?",                    "genre_filter"),
    ("Lọc toàn bộ phim {GENRE} liên quan đến {DIRECTOR}",                  "genre_filter"),
    ("Danh sách phim {GENRE} có đánh giá cao năm {YEAR}",                  "genre_filter"),
    ("Tìm phim {GENRE} hợp xem gia đình",                                  "genre_filter"),
    ("Cho tôi phim {GENRE} nổi bật nhất của {ACTOR}",                      "genre_filter"),
    ("Lọc những phim {GENRE} đáng xem của {DIRECTOR}",                     "genre_filter"),
    ("Hiển thị phim {GENRE} từ năm {YEAR} đến nay",                        "genre_filter"),
    ("Có các phim {GENRE} nào gắn với {MOVIE_TITLE}?",                     "genre_filter"),
    ("Tìm phim {GENRE} có dàn cast gồm {ACTOR}",                           "genre_filter"),
    ("Cho tôi xem phim {GENRE} theo từng năm quanh {YEAR}",                "genre_filter"),
    ("Danh sách phim {GENRE} của {DIRECTOR} được khen nhiều",              "genre_filter"),
    ("Lọc phim {GENRE} có màu sắc giống {MOVIE_TITLE}",                    "genre_filter"),
    ("Tìm phim {GENRE} có doanh thu tốt năm {YEAR}",                       "genre_filter"),
    ("Hiển thị phim {GENRE} có {ACTOR} đóng phụ cũng được",                "genre_filter"),
    ("Tôi muốn danh mục phim {GENRE} nổi theo năm {YEAR}",                 "genre_filter"),
    ("Lọc phim {GENRE} mà {DIRECTOR} làm gần đây",                         "genre_filter"),
    ("Danh sách phim {GENRE} có thời lượng vừa phải",                      "genre_filter"),
    ("Tìm phim {GENRE} có phong cách như {MOVIE_TITLE}",                   "genre_filter"),
    ("Cho tôi phim {GENRE} đáng chú ý của {ACTOR}",                        "genre_filter"),
    ("Lọc phim {GENRE} phát hành đúng năm {YEAR}",                         "genre_filter"),
    ("Có phim {GENRE} nào của {DIRECTOR} hợp người mới xem không?",        "genre_filter"),
    ("Hiển thị phim {GENRE} được tìm nhiều nhất",                          "genre_filter"),
    ("Tìm phim {GENRE} có yếu tố thương mại mạnh",                         "genre_filter"),
    ("Cho tôi danh sách phim {GENRE} nhiều giải thưởng",                   "genre_filter"),
    ("Lọc phim {GENRE} theo đạo diễn {DIRECTOR} và năm {YEAR}",            "genre_filter"),
    ("Có những phim {GENRE} nào liên quan tới {ACTOR}?",                   "genre_filter"),
    ("Tìm phim {GENRE} kiểu đại chúng phát hành năm {YEAR}",               "genre_filter"),
    ("Hiển thị top phim {GENRE} có {ACTOR} trong cast",                    "genre_filter"),
    ("Lọc phim {GENRE} xem ổn của {DIRECTOR}",                            "genre_filter"),
    ("Danh sách phim {GENRE} dễ recommend nhất",                           "genre_filter"),
    ("Tìm phim {GENRE} ra mắt quanh năm {YEAR}",                           "genre_filter"),
    ("Cho tôi xem phim {GENRE} bị đánh giá thấp oan",                      "genre_filter"),
    ("Lọc phim {GENRE} cùng tông với {MOVIE_TITLE}",                       "genre_filter"),
    ("Có phim {GENRE} nào của {ACTOR} doanh thu cao không?",               "genre_filter"),
    ("Hiển thị phim {GENRE} do {DIRECTOR} cầm trịch gần đây",              "genre_filter"),
    ("Tìm phim {GENRE} được khán giả trẻ thích",                           "genre_filter"),
    ("Danh sách phim {GENRE} tôi nên xem trước tiên",                      "genre_filter"),
    ("Lọc phim {GENRE} có lượt quan tâm cao năm {YEAR}",                   "genre_filter"),
    ("Cho xem các phim {GENRE} tiêu biểu của {DIRECTOR}",                  "genre_filter"),
    ("Tìm phim {GENRE} hợp gu nếu thích {MOVIE_TITLE}",                    "genre_filter"),
    ("Có phim {GENRE} nào của {ACTOR} ra sau {YEAR}?",                     "genre_filter"),
    ("Hiển thị phim {GENRE} theo mức độ phổ biến",                         "genre_filter"),
    ("Tìm phim {GENRE} được đánh giá ổn định qua thời gian",               "genre_filter"),
    ("Danh sách phim {GENRE} đáng chú ý nhất quanh {YEAR}",                "genre_filter"),
    ("Lọc phim {GENRE} có mặt {ACTOR} và tên tuổi {DIRECTOR}",             "genre_filter"),
    ("Cho tôi xem các phim {GENRE} đáng bàn luận gần đây",                 "genre_filter"),
]

_NER_TEMPLATE_COUNTS = {}
for _, intent in _NER_TEMPLATES:
    _NER_TEMPLATE_COUNTS[intent] = _NER_TEMPLATE_COUNTS.get(intent, 0) + 1

for _intent in ["find_movie", "person_info", "movie_info", "recommendation", "genre_filter"]:
    if _NER_TEMPLATE_COUNTS.get(_intent, 0) != 100:
        raise ValueError(
            f"_NER_TEMPLATES for {_intent} must be 100, got {_NER_TEMPLATE_COUNTS.get(_intent, 0)}"
        )

# ============================================================
# NEGATIVE NER TEMPLATES (P0: cau KHONG co entity nao)
# ============================================================
_NER_NEGATIVE_TEMPLATES = [
    ("Tôi muốn tìm phim hay",                                              "find_movie"),
    ("Tìm cho tôi một bộ phim mới xem nhé",                                "find_movie"),
    ("Có phim nào đáng xem không?",                                        "find_movie"),
    ("Giúp mình tìm phim với",                                             "find_movie"),
    ("Tìm phim mới nhất đi",                                               "find_movie"),
    ("Gợi ý phim hay đi",                                                  "recommendation"),
    ("Cho tôi vài gợi ý phim đáng xem",                                    "recommendation"),
    ("Hôm nay xem phim gì hay?",                                           "recommendation"),
    ("Đề xuất phim hay cho cuối tuần",                                     "recommendation"),
    ("Tôi đang buồn, gợi ý phim gì vui đi",                                "recommendation"),
    ("Phim nào đáng xem hiện tại?",                                        "recommendation"),
    ("Cho mình xem phim hay nhất năm nay",                                 "recommendation"),
    ("Phim đó nội dung gì vậy?",                                           "movie_info"),
    ("Cho tôi biết thông tin phim đó",                                     "movie_info"),
    ("Phim đó có hay không?",                                              "movie_info"),
    ("Ai đóng vai chính trong phim đó?",                                   "movie_info"),
    ("Phim đó dài bao lâu?",                                               "movie_info"),
    ("Diễn viên đó có phim gì hay không?",                                 "person_info"),
    ("Cho tôi xem danh sách phim của diễn viên đó",                        "person_info"),
    ("Tìm phim theo thể loại đi",                                          "genre_filter"),
    ("Lọc phim hay nhất cho tôi",                                          "genre_filter"),
    ("Cho tôi danh sách phim đang chiếu",                                  "genre_filter"),
    ("Hiện tại phim nào đang hot nhất?",                                   "genre_filter"),
    ("Danh sách phim mới ra tháng này",                                    "genre_filter"),
    ("Phim nào đang được yêu thích nhất?",                                 "genre_filter"),
    # --- Chat intents (khong co slot) ---
    ("Xin chào",                                                            "greeting"),
    ("Chào bạn",                                                            "greeting"),
    ("Hello",                                                               "greeting"),
    ("Hi bạn",                                                              "greeting"),
    ("Chào buổi sáng",                                                      "greeting"),
    ("Alo chatbot ơi",                                                      "greeting"),
    ("Chào chatbot",                                                        "greeting"),
    ("Mình cần tìm phim, chào bạn",                                        "greeting"),
    ("Chatbot ơi, cho mình hỏi",                                           "greeting"),
    ("Xin chào, bạn có thể giúp mình không?",                              "greeting"),
    ("Bắt đầu tìm kiếm phim nào",                                         "greeting"),
    ("Hi chatbot",                                                          "greeting"),
    ("Hey bạn, mình cần gợi ý phim",                                       "greeting"),
    ("Chào nha",                                                            "greeting"),
    ("Hello bạn ơi",                                                        "greeting"),
    ("Tạm biệt",                                                           "goodbye"),
    ("Bye bye",                                                             "goodbye"),
    ("Cảm ơn nhé",                                                         "goodbye"),
    ("Hẹn gặp lại",                                                        "goodbye"),
    ("OK xong rồi cảm ơn",                                                 "goodbye"),
    ("Cảm ơn bạn nhiều lắm, tạm biệt",                                    "goodbye"),
    ("Mình tìm được rồi, cảm ơn",                                         "goodbye"),
    ("Bye nha",                                                             "goodbye"),
    ("Tạm biệt bạn",                                                       "goodbye"),
    ("Cảm ơn nhiều, bye",                                                  "goodbye"),
    ("OK mình đi đây",                                                     "goodbye"),
    ("Xong rồi, cảm ơn bạn",                                              "goodbye"),
    ("Thanks nhé",                                                          "goodbye"),
    ("Mình hài lòng rồi, tạm biệt",                                       "goodbye"),
    ("Đủ rồi, bye bye",                                                    "goodbye"),
    ("Hôm nay thời tiết thế nào?",                                         "out_of_scope"),
    ("Bạn là AI gì vậy?",                                                  "out_of_scope"),
    ("Giá vé xem phim bao nhiêu?",                                         "out_of_scope"),
    ("Kể chuyện cười đi",                                                  "out_of_scope"),
    ("2 + 2 bằng bao nhiêu",                                               "out_of_scope"),
    ("Dạy tôi nấu phở đi",                                                "out_of_scope"),
    ("Bạn có thể đặt vé máy bay không?",                                   "out_of_scope"),
    ("Tôi muốn nghe nhạc",                                                 "out_of_scope"),
    ("Bạn tên gì?",                                                        "out_of_scope"),
    ("Mấy giờ rồi?",                                                       "out_of_scope"),
    ("Dạy mình code Python đi",                                            "out_of_scope"),
    ("Cho mình số điện thoại của rạp CGV",                                 "out_of_scope"),
    ("Đặt pizza giúp mình",                                                "out_of_scope"),
    ("Cho mình xem lịch chiếu rạp",                                        "out_of_scope"),
    ("Bạn có biết hát không?",                                             "out_of_scope"),
]

# ============================================================
# GENERATE NER BIO DATA (Character-level spans cho BIO tagging)
# ============================================================

def _entity_substitution_augment(sample: dict, vocab: dict, n_augments: int = 2) -> list:
    """Entity substitution augmentation cho NER BIO data.
    Thay the entity cu bang entity moi cung loai, cap nhat span offsets."""
    augmented = []
    text = sample["text"]
    entity_spans = sample["entity_spans"]
    intent = sample["intent"]
    
    if not entity_spans:
        return []
    
    for _ in range(n_augments):
        new_text = text
        new_spans = []
        offset_delta = 0
        sorted_spans = sorted(entity_spans, key=lambda x: x[0])
        
        for orig_start, orig_end, label in sorted_spans:
            adj_start = orig_start + offset_delta
            adj_end = orig_end + offset_delta
            old_val = new_text[adj_start:adj_end]
            
            if label == "PERSON":
                candidates = vocab.get("_persons", [])
            elif label == "MOVIE_TITLE":
                candidates = vocab.get("movies", [])
            elif label == "GENRE":
                candidates = vocab.get("genres", [])
            elif label == "YEAR":
                candidates = [str(y) for y in range(2010, 2025)]
            else:
                candidates = []
            
            if candidates:
                new_val = random.choice([c for c in candidates if c != old_val] or candidates)
            else:
                new_val = old_val
            
            new_text = new_text[:adj_start] + new_val + new_text[adj_end:]
            new_spans.append((adj_start, adj_start + len(new_val), label))
            offset_delta += len(new_val) - len(old_val)
        
        augmented.append({
            "text": new_text,
            "entity_spans": new_spans,
            "intent": intent,
            "source": "entity_sub_augmented",
        })
    
    return augmented


def generate_ner_bio_data(
    entities: dict,
    n_samples: int = 800,
    n_entity_sub_augments: int = 2,
) -> list:
    """Sinh du lieu NER voi character-level entity spans cho BIO tagging.
    Bao gom:
    - Template filling voi uniform sampling
    - Entity substitution augmentation (an toan vi giu nguyen span structure)
    - Negative samples (cau khong co entity)
    """
    all_actors    = entities.get("actors",    ["Thành Long"])
    all_directors = entities.get("directors", ["James Cameron"])
    movies        = entities.get("movies",    ["Avatar"])
    genres        = entities.get("genres",    ["Hành Động"])
    years         = [str(y) for y in range(2010, 2025)]

    random.shuffle(all_actors)
    random.shuffle(all_directors)
    random.shuffle(movies)

    actor_set    = set(all_actors)
    director_set = set(all_directors)
    overlap_set  = actor_set & director_set
    pure_actors    = [a for a in all_actors    if a not in overlap_set] or all_actors
    pure_directors = [d for d in all_directors if d not in overlap_set] or all_directors
    
    _sub_vocab = {
        "_persons": list(set(all_actors + all_directors)),
        "movies": movies,
        "genres": genres,
    }

    _vocab_bio = {
        "{MOVIE_TITLE}": (movies, "MOVIE_TITLE"),
        "{GENRE}":       (genres, "GENRE"),
        "{YEAR}":        (years,  "YEAR"),
    }

    _REPLACE_ORDER = ["{ACTOR}", "{DIRECTOR}", "{MOVIE_TITLE}", "{GENRE}", "{YEAR}"]

    dataset = []
    
    for _ in range(n_samples):
        tpl, intent = random.choice(_NER_TEMPLATES)

        replacements = {}
        if "{ACTOR}" in tpl:
            val = _get_uniform_choice(pure_actors)
            if random.random() < 0.3:
                val = val.lower()
            replacements["{ACTOR}"] = (val, "PERSON")
        if "{DIRECTOR}" in tpl:
            val = _get_uniform_choice(pure_directors)
            if replacements.get("{ACTOR}", ("",))[0] == val:
                val = _get_uniform_choice(pure_directors)
            if random.random() < 0.3:
                val = val.lower()
            replacements["{DIRECTOR}"] = (val, "PERSON")
        for ph, (vocab, label) in _vocab_bio.items():
            if ph in tpl:
                val = _get_uniform_choice(vocab)
                if label == "GENRE":
                    if random.random() < 0.3 and val in _GENRE_SYNONYMS:
                        val = random.choice(_GENRE_SYNONYMS[val])
                    else:
                        val = random.choice([val, val.lower(), val.title()])
                elif label == "MOVIE_TITLE" and random.random() < 0.3:
                    val = val.lower()
                replacements[ph] = (val, label)

        text = tpl
        entity_spans = []
        for ph in _REPLACE_ORDER:
            if ph in replacements and ph in text:
                val, label = replacements[ph]
                idx = text.find(ph)
                text = text[:idx] + val + text[idx + len(ph):]
                entity_spans.append((idx, idx + len(val), label))

        sample = {
            "text": text,
            "entity_spans": entity_spans,
            "intent": intent,
            "source": "template",
        }
        dataset.append(sample)
        
        aug_samples = _entity_substitution_augment(sample, _sub_vocab, n_augments=n_entity_sub_augments)
        dataset.extend(aug_samples)

    n_negative = max(int(n_samples * 0.1), len(_NER_NEGATIVE_TEMPLATES))
    for _ in range(n_negative):
        tpl, intent = random.choice(_NER_NEGATIVE_TEMPLATES)
        dataset.append({
            "text": tpl,
            "entity_spans": [],
            "intent": intent,
            "source": "negative",
        })

    random.shuffle(dataset)
    print(f"OK NER BIO data: {len(dataset)} samples (positive: ~{n_samples*(1+n_entity_sub_augments)}, negative: ~{n_negative})")

    n_check = min(20, len(dataset))
    errors = 0
    for sample in dataset[:n_check]:
        for start, end, label in sample["entity_spans"]:
            extracted = sample["text"][start:end]
            if not extracted.strip():
                errors += 1
    if errors:
        print(f"   [!] {errors}/{n_check} samples co span rong -> kiem tra lai template!")
    else:
        print(f"   Span validation OK ({n_check} samples checked)")

    return dataset



# ============================================================
# GENERATE FRAME DATA (Semantic Parsing format)
# ============================================================

def _entity_substitution_augment_frame(sample: dict, vocab: dict, n_augments: int = 2) -> list:
    """Entity substitution augmentation cho frame data.
    Thay entity cu bang entity moi cung loai, cap nhat argument spans."""
    augmented = []
    text = sample["text"]
    arguments = sample.get("arguments", {})
    intent = sample["intent"]
    frame = sample["frame"]

    if not arguments:
        return []

    for _ in range(n_augments):
        new_text = text
        new_args = {}
        offset_delta = 0

        sorted_args = sorted(arguments.items(), key=lambda x: x[1]["start"])

        for slot_name, arg in sorted_args:
            adj_start = arg["start"] + offset_delta
            adj_end = arg["end"] + offset_delta
            old_val = new_text[adj_start:adj_end]
            entity_type = SLOT_TO_ENTITY.get(slot_name, "")

            if entity_type == "PERSON":
                candidates = vocab.get("_persons", [])
            elif entity_type == "MOVIE_TITLE":
                candidates = vocab.get("movies", [])
            elif entity_type == "GENRE":
                candidates = vocab.get("genres", [])
            elif entity_type == "YEAR":
                candidates = [str(y) for y in range(2010, 2025)]
            else:
                candidates = []

            if candidates:
                new_val = random.choice([c for c in candidates if c != old_val] or candidates)
            else:
                new_val = old_val

            new_text = new_text[:adj_start] + new_val + new_text[adj_end:]
            new_args[slot_name] = {
                "value": new_val,
                "start": adj_start,
                "end": adj_start + len(new_val),
            }
            offset_delta += len(new_val) - len(old_val)

        augmented.append({
            "text": new_text,
            "intent": intent,
            "label_id": LABEL2ID.get(intent, 0),
            "frame": frame,
            "arguments": new_args,
            "source": "entity_sub_augmented",
        })

    return augmented


def generate_frame_data(
    entities: dict,
    n_samples: int = 800,
    n_entity_sub_augments: int = 2,
) -> list:
    """Sinh du lieu frame-annotated cho Semantic Parsing.
    Output: list of {text, intent, label_id, frame, arguments: {slot: {value, start, end}}, source}
    """
    all_actors    = entities.get("actors",    ["Thanh Long"])
    all_directors = entities.get("directors", ["James Cameron"])
    movies        = entities.get("movies",    ["Avatar"])
    genres        = entities.get("genres",    ["Hanh Dong"])
    years         = [str(y) for y in range(2010, 2025)]

    random.shuffle(all_actors)
    random.shuffle(all_directors)
    random.shuffle(movies)

    actor_set    = set(all_actors)
    director_set = set(all_directors)
    overlap_set  = actor_set & director_set
    pure_actors    = [a for a in all_actors if a not in overlap_set] or all_actors
    pure_directors = [d for d in all_directors if d not in overlap_set] or all_directors

    _sub_vocab = {
        "_persons": list(set(all_actors + all_directors)),
        "movies": movies,
        "genres": genres,
    }

    _REPLACE_ORDER = ["{ACTOR}", "{DIRECTOR}", "{MOVIE_TITLE}", "{GENRE}", "{YEAR}"]

    dataset = []

    for _ in range(n_samples):
        tpl, intent = random.choice(_NER_TEMPLATES)
        frame_info = FRAME_SCHEMA.get(intent, {})
        frame_name = frame_info.get("frame", intent.upper())
        slot_map = _PH_TO_SLOT.get(intent, {})

        replacements = {}
        if "{ACTOR}" in tpl:
            val = _get_uniform_choice(pure_actors)
            if random.random() < 0.3:
                val = val.lower()
            replacements["{ACTOR}"] = val
        if "{DIRECTOR}" in tpl:
            val = _get_uniform_choice(pure_directors)
            if replacements.get("{ACTOR}") == val:
                val = _get_uniform_choice(pure_directors)
            if random.random() < 0.3:
                val = val.lower()
            replacements["{DIRECTOR}"] = val
        if "{MOVIE_TITLE}" in tpl:
            val = _get_uniform_choice(movies)
            if random.random() < 0.3:
                val = val.lower()
            replacements["{MOVIE_TITLE}"] = val
        if "{GENRE}" in tpl:
            g = _get_uniform_choice(genres)
            # 30% dùng synonym thay vì genre chuẩn
            if random.random() < 0.3 and g in _GENRE_SYNONYMS:
                g = random.choice(_GENRE_SYNONYMS[g])
            else:
                g = random.choice([g, g.lower(), g.title()])
            replacements["{GENRE}"] = g
        if "{YEAR}" in tpl:
            replacements["{YEAR}"] = _get_uniform_choice(years)

        text = tpl
        arguments = {}
        for ph in _REPLACE_ORDER:
            if ph in replacements and ph in text:
                val = replacements[ph]
                idx = text.find(ph)
                text = text[:idx] + val + text[idx + len(ph):]

                slot_name = slot_map.get(ph)
                if slot_name:
                    arguments[slot_name] = {
                        "value": val,
                        "start": idx,
                        "end": idx + len(val),
                    }

        sample = {
            "text": text,
            "intent": intent,
            "label_id": LABEL2ID.get(intent, 0),
            "frame": frame_name,
            "arguments": arguments,
            "source": "template",
        }
        dataset.append(sample)

        if n_entity_sub_augments > 0 and arguments:
            augmented = _entity_substitution_augment_frame(
                sample, _sub_vocab, n_entity_sub_augments
            )
            dataset.extend(augmented)

    # Negative samples (cau khong co argument nao)
    # Cap repetition de tranh overfit vao exact text
    neg_repeat = min(max(n_samples // len(_NER_NEGATIVE_TEMPLATES), 5), 30)
    for tpl, intent in _NER_NEGATIVE_TEMPLATES:
        frame_info = FRAME_SCHEMA.get(intent, {})
        frame_name = frame_info.get("frame", intent.upper())
        for _ in range(neg_repeat):
            dataset.append({
                "text": tpl,
                "intent": intent,
                "label_id": LABEL2ID.get(intent, 0),
                "frame": frame_name,
                "arguments": {},
                "source": "negative",
            })

    random.shuffle(dataset)
    print(f"OK Frame data: {len(dataset)} samples")
    return dataset


def save_dataset(dataset: list, path: str = None):
    path = path or os.path.join(DATA_PROCESSED, "intent_dataset.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(dataset)} câu -> {path}")

save_intent_data = save_dataset
