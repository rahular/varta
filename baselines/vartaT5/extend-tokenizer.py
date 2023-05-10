# need to run this in sentencepiece/python/src/sentencepiece/
import sys
import sentencepiece_model_pb2 as model
m = model.ModelProto()

spm = sys.argv[1]

m.ParseFromString(open(f"/path/to/spm/model/{spm}.model", "rb").read())

special_tokens = [f"<extra_id_{i}>" for i in range(99, -1, -1)]

for token in special_tokens:
    new_token = model.ModelProto().SentencePiece()
    new_token.piece = token
    new_token.score = 0
    m.pieces.append(new_token)

with open(f'/path/to/updated/spm/model/updated{spm}.model', 'wb') as f:
    f.write(m.SerializeToString())