import vncorenlp
import os

# Set absolute path where the VnCoreNLP model will be saved and loaded from
vncorenlp_path = os.path.abspath('./vncorenlp')  # <-- change this to a real directory if needed

# Step 1: Automatically download VnCoreNLP components (only needs to be done once)
vncorenlp.download_model(save_dir=vncorenlp_path)

# Step 2: Load VnCoreNLP with all annotators
model = vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=vncorenlp_path)

# Step 3: Annotate raw text (you can skip annotate_file if you work with Python data)
text = "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
annotations = model.annotate_text(text)

# Step 4: Print annotation output
model.print_out(annotations)
