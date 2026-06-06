**Toàn bộ cuộc trò chuyện**

_Chủ đề: Paper Video Retrieval / Video Browser Showdown 2025_

| **File paper được nhắc tới** | MultiMedia Modeling 31st International Conference on Multimedia Modeling, MMM 2025, Nara, Japan.pdf  |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Nội dung chính**           | Lọc ra các điểm hay nhất trong hệ thống NII-UIT at VBS2025 để học hỏi và triển khai video retrieval. |
| **Ghi chú**                  | Tài liệu này ghi lại phần trao đổi chính trong cuộc trò chuyện hiện tại.                             |

**Người dùng**

Đọc paper này hãy lọc ra những gì hay nhất mà nhóm này đã làm để có thể vô địch cuộc thi. Lọc ra những thứ hay nhất để tui học hỏi là thực hiện theo. Cái này là về Video Retrival

**ChatGPT**

Đúng paper này là NII-UIT at VBS2025: Multimodal Video Retrieval with LLM Integration and Dynamic Temporal Search. Mình có kiểm tra thêm ngoài paper: UIT công bố đội NII_UIT giành giải Nhất VBS 2025, còn paper này mô tả hệ thống kỹ thuật của họ.

Điểm hay nhất không nằm ở "dùng model siêu to" đơn lẻ, mà là họ ghép nhiều cách tìm kiếm lại thành một hệ thống tương tác rất nhanh, rất linh hoạt.

1\. Cái đáng học nhất: họ không chỉ search bằng text

Hệ thống của họ hỗ trợ nhiều kiểu query cùng lúc:

\- text query

\- visual query

\- object filtering

\- query expansion bằng LLM

\- sinh ảnh bằng Stable Diffusion để làm visual query

\- temporal search theo nhiều giai đoạn

Trong abstract, họ nói rõ framework hỗ trợ text queries, visual data, object filtering, visual queries generated with Stable Diffusion, và có dynamic temporal search để đánh giá độ liên quan theo thời gian tốt hơn cách truyền thống.

Nói đơn giản:

Query gốc → LLM paraphrase / synonym / enrich context → CLIP / VLM text-image search → Stable Diffusion sinh ảnh minh họa rồi search bằng ảnh → object detector lọc vật thể → temporal search tìm quanh đoạn trước/sau → fusion score → UI cho người dùng refine tiếp.

Đây là ý rất đáng bắt chước.

2\. Offline xử lý thông minh: không index mọi frame

Họ không xử lý toàn bộ frame vì dataset VBS rất lớn. Paper nói rõ xử lý mọi frame là kém hiệu quả, tốn thời gian, tốn lưu trữ và làm chậm truy vấn. Vì vậy họ chọn keyframe selection: lấy đặc trưng semantic bằng BEiT-3 từ mỗi frame thứ 10, rồi chỉ giữ các frame có khác biệt đáng kể. Keyframe còn được lưu dạng WebP để giảm dung lượng nhưng vẫn giữ chất lượng.

Cái này cực quan trọng cho project của ông.

Ông không nên làm kiểu:

Video → cắt tất cả frame → embedding tất cả

Nên làm:

Video → shot/scene segmentation → lấy keyframe đại diện → loại frame gần trùng → embedding keyframe

Nếu muốn đơn giản hơn NII-UIT, ông có thể bắt đầu bằng:

1 frame / 1 giây + loại frame gần giống nhau bằng cosine similarity hoặc perceptual hash + lưu thumbnail WebP.

Cái này tăng tốc rất nhiều.

3\. Vector database + tabular database tách riêng

Trong pipeline của họ, offline stage gồm:

keyframe → feature vector → Milvus vector database

keyframe → object detection → tabular database

Paper nêu rõ feature vectors được lưu trong Milvus, còn object detection được lưu trong database dạng bảng.

Đây là thiết kế rất hay vì mỗi loại dữ liệu có cách tìm khác nhau:

Embedding / semantic search → vector DB

Object count / object name / position → SQL / tabular DB

OCR / caption / transcript → text index

Với project của ông, nên thiết kế kiểu này:

Milvus / FAISS:

\- image_embedding

\- text_embedding

PostgreSQL / SQLite:

\- video_id

\- shot_id

\- timestamp

\- object_name

\- object_count

\- OCR text

\- caption

\- transcript

Đừng cố nhét mọi thứ vào một chỗ.

4\. Fusion nhiều model thay vì tin một model

Họ dùng các Vision-Language Models tiên tiến như BEiT-3, OpenCLIP H-14, InternVL-G để tăng chất lượng search text-visual. Kết quả từ từng model được normalize rồi fusion lại bằng thuật toán kết hợp điểm, nhằm cân bằng điểm mạnh/yếu của từng model.

Đây là bài học lớn: Một model có thể giỏi cảnh tổng quát, model khác giỏi object, model khác giỏi caption/semantic. Người thắng thường không dùng một đường duy nhất.

Công thức thực tế ông có thể làm:

final_score = 0.45 \* CLIP_score + 0.25 \* caption_score + 0.15 \* OCR_score + 0.10 \* object_score + 0.05 \* transcript_score

Sau đó cho người dùng chỉnh weight trong UI như họ làm ở Advanced Mode.

5\. Query expansion bằng LLM là vũ khí mạnh

Paper nói người dùng thường query ngắn, mơ hồ hoặc thiếu thông tin, làm kết quả kém. Họ dùng GPT-4o để sinh nhiều phiên bản paraphrase của query gốc, rồi cho người dùng chọn hoặc chạy song song các query này.

Ví dụ user gõ:

man fixing car

LLM có thể mở rộng thành:

a person repairing a vehicle

a mechanic working on a car

a man using tools near an automobile

a person checking the engine of a car

Cái này giúp tăng recall cực mạnh, nhất là khi caption/embedding không khớp đúng từ người dùng gõ.

Với hệ thống của ông, nên làm module:

Query → LLM rewrite thành 5 query → search từng query → merge kết quả → rerank.

6\. Stable Diffusion để biến text thành ảnh rồi search ảnh

Đây là một chiêu rất hay. Paper nói họ dùng Stable Diffusion để sinh ảnh từ mô tả text, rồi dùng ảnh đó như visual query. Việc này giúp bổ sung góc nhìn hình ảnh cho query văn bản.

Ví dụ query:

a red car in a showroom

Họ có thể sinh vài ảnh "red car in showroom", rồi lấy embedding ảnh sinh ra để search keyframe gần giống.

Cái này hay vì đôi khi text embedding không đủ tốt, nhưng image-to-image similarity lại bắt màu sắc, bố cục, cảnh tốt hơn.

Tuy nhiên với project của ông, cái này nên để phase 2, vì khá nặng. Làm sau khi đã có CLIP + caption + object filter.

7\. Object filtering giải quyết điểm yếu của embedding

Embedding-based search giỏi tìm "ý nghĩa chung", nhưng yếu khi cần đếm vật thể hoặc phân biệt vật thể cụ thể. Paper nói embedding model gặp khó với object counting và object distinction trong cảnh phức tạp, nên họ tích hợp Co-DETR để detect/count object COCO trong từng keyframe. Sau đó dùng object data để lọc keyframe theo query.

Ví dụ query:

two dogs near a person

CLIP có thể trả về ảnh có một con chó hoặc nhiều người, không chắc đúng. Object filter sẽ giúp lọc:

dog_count >= 2

person_count >= 1

Với project của ông, có thể dùng YOLO trước cho nhẹ:

YOLOv8 / YOLOv11 → object_name, confidence, bbox

Sau đó query có thể lọc:

object = "car"

object_count >= 2

position = left/right/center

Đây là module nên làm sớm.

8\. Dynamic temporal search là điểm rất đáng học

Đây có thể là phần "ăn điểm" nhất.

Với Known-Item Search dạng mô tả nhiều đoạn, người dùng có thể nói:

First, a man is walking in a street. Then, he enters a shop. Later, he talks to a woman.

Cách thường là search từng câu rồi ép thứ tự trước/sau. Nhưng paper nói quan hệ thời gian đôi khi không rõ, nên họ không chỉ xét "trước" hoặc "sau". Họ tìm các shot xung quanh kết quả ban đầu, đánh giá độ liên quan của các shot lân cận với query mới, rồi rerank bằng cách cộng điểm qua nhiều stage.

Cách hiểu đơn giản:

Stage 1 search: "man walking in street" → lấy top result tại timestamp T

Stage 2 search: "enters a shop" → không search toàn bộ dataset từ đầu → tìm quanh T: T-30s đến T+30s → rerank các shot gần đó

Stage 3 search: "talks to a woman" → tiếp tục tìm quanh vùng kết quả tốt nhất

Cái này cực hợp với Video Retrieval vì video có tính liên tục. Nếu ông muốn tăng độ chính xác, module này nên ưu tiên cao.

9\. UI của họ không chỉ đẹp, mà hỗ trợ tốc độ thao tác

Paper mô tả UI có hai chế độ hiển thị:

\- nhóm các shot gần nhau để dễ so sánh

\- hiển thị top-scoring frame của mỗi shot để tìm nhanh

Advanced Mode cho phép chỉnh model weights, paraphrase query, sinh visual query từ text; còn novice mode thì tắt bớt để người mới không bị rối.

Đây là bài học quan trọng: VBS không chỉ chấm model, mà còn chấm khả năng người dùng tìm đúng nhanh.

Với UI của ông, nên có:

Search box + query expansion button + object filter + timeline around result + similar frame search + grouped shots + top frame per video/shot + adjustable weight advanced mode.

10\. Thứ đáng copy nhất cho project của ông

Ưu tiên 1: Keyframe selection + WebP thumbnail - tăng tốc, giảm lưu trữ.

Ưu tiên 2: CLIP/OpenCLIP embedding search - xương sống của video retrieval.

Ưu tiên 3: Caption/OCR/Object/Transcript index riêng - search đa nguồn, không phụ thuộc một model.

Ưu tiên 4: Query expansion bằng LLM - tăng recall rất mạnh.

Ưu tiên 5: Score fusion nhiều nguồn - kết quả ổn định hơn.

Ưu tiên 6: Object filtering bằng YOLO/Co-DETR - bắt chi tiết vật thể, số lượng.

Ưu tiên 7: Dynamic temporal search - tìm đúng đoạn liên tục trong video.

Ưu tiên 8: UI có novice/advanced mode - người dùng thao tác nhanh hơn.

Ưu tiên 9: Stable Diffusion visual query - hay nhưng nên làm sau.

Ưu tiên 10: Cho chỉnh weight model - phù hợp VBS/interactive search.

Kết luận ngắn gọn

Nhóm này thắng không phải vì một module "ma thuật", mà vì họ làm đúng bản chất của Video Retrieval:

Tìm bằng nhiều nguồn + hiểu query bằng LLM + lọc bằng object + tìm theo thời gian + fusion điểm + UI giúp người dùng refine rất nhanh.

Với project của ông, mình nghĩ nên học theo hướng này:

Phase 1: Video → scene/keyframe → CLIP embedding → caption → OCR → YOLO object → index → web search UI

Phase 2: LLM query expansion → score fusion → object filter → similar search

Phase 3: Dynamic temporal search → grouped shots → advanced mode chỉnh weight

Phase 4: Stable Diffusion visual query → rerank nâng cao

Cái nên làm trước nhất là CLIP + caption + object filter + temporal search, vì đây là combo vừa thực tế vừa dễ chứng minh với thầy là hệ thống có cải tiến rõ ràng.

**Người dùng**

Xuất word tất cả cuộc trò chuyện này