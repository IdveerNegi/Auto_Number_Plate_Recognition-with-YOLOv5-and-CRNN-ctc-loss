
from flask import Flask, render_template, request, url_for, Response
from flask_restful import Api, Resource, reqparse
import pytesseract
import cv2
from PIL import Image
import os, werkzeug
from math import floor
import base64
import sys
sys.path.append("src")
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from utils import alphabet
import models.crnn_lang as crnn
from pathlib import Path
use_gpu = True

encoder_path = './expr/attentioncnn/encoder_bt.pth'
decoder_path = './expr/attentioncnn/decoder_bt.pth'

max_length = 15                          # 最长字符串的长度
EOS_TOKEN = 1

nclass = len(alphabet) + 3
encoder = crnn.CNN(32, 1, 256)          # 编码器
# decoder = crnn.decoder(256, nclass)     # seq to seq的解码器, nclass在decoder中还加了2
decoder = crnn.decoderV2(256, nclass)


if encoder_path and decoder_path:
    print('loading pretrained models ......')
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
if torch.cuda.is_available() and use_gpu:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

image_folder = "correct/"
converter = utils.strLabelConverterForAttention(alphabet)
for img in os.listdir(image_folder):
    img_path = image_folder + img
    transformer = dataset.resizeNormalize((280, 32))
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available() and use_gpu:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    encoder.eval()
    decoder.eval()
    encoder_out = encoder(image)

    decoded_words = []
    prob = 1.0
    decoder_attentions = torch.zeros(max_length, 71)
    decoder_input = torch.zeros(1).long()      # 初始化decoder的开始,从0开始输出
    decoder_hidden = decoder.initHidden(1)
    if torch.cuda.is_available() and use_gpu:
        decoder_input = decoder_input.cuda()
        decoder_hidden = decoder_hidden.cuda()
    loss = 0.0
    # 预测的时候采用非强制策略，将前一次的输出，作为下一次的输入，直到标签为EOS_TOKEN时停止
    for di in range(max_length):  # 最大字符串的长度
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_out)
        probs = torch.exp(decoder_output)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoder_input = ni
        prob *= probs[:, ni]
        if ni == EOS_TOKEN:
            # decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(converter.decode(ni))

    words = ''.join(decoded_words)
    prob = prob.item()
    name = Path(img_path).name
    print(name)
    print('predict_str:%-20s => prob:%-20s' % (words, prob))


REDUCTION_COEFF = 0.9
QUALITY = 85

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')


@app.route('/')
def home():
    return render_template('index.html')
    


@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    try:
        imagefile = request.files.get('imagefile', '')
        #create byte stream from uploaded file
        file = request.files['imagefile'].read() ## byte file
        converter = utils.strLabelConverterForAttention(alphabet)
        img = Image.open(imagefile)
        transformer = dataset.resizeNormalize((280, 32))
        image = Image.open(imagefile).convert('L')
        image = transformer(image)
        if torch.cuda.is_available() and use_gpu:
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        encoder.eval()
        decoder.eval()
        encoder_out = encoder(image)
        decoded_words = []
        prob = 1.0
        decoder_attentions = torch.zeros(max_length, 71)
        decoder_input = torch.zeros(1).long()      # 初始化decoder的开始,从0开始输出
        decoder_hidden = decoder.initHidden(1)
        if torch.cuda.is_available() and use_gpu:
            decoder_input = decoder_input.cuda()
            decoder_hidden = decoder_hidden.cuda()
        loss = 0.0
        # 预测的时候采用非强制策略，将前一次的输出，作为下一次的输入，直到标签为EOS_TOKEN时停止
        for di in range(max_length):  # 最大字符串的长度
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_out)
            probs = torch.exp(decoder_output)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            prob *= probs[:, ni]
            if ni == EOS_TOKEN:
                # decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(converter.decode(ni))

        words = ''.join(decoded_words)
        prob = prob.item()
        # print('predict_str:%-20s => prob:%-20s' % (words, prob))
        img_base64 = base64.b64encode(file)
        img_base64_str = str(img_base64)
        #final base64 encoded string
        img_base64_str = "data:image/"+ words +";base64,"+img_base64_str.split('\'',1)[1][0:-1]
        f = open("sample.txt", "a")
        f.truncate(0)
        f.write(words)
        f.close()
        return render_template('result.html', var=words,img=img_base64_str)
    except Exception as e:
        print(e) 
        return render_template('error.html')
    
@app.route("/gettext")
def gettext():
        with open("sample.txt") as fp:
            src = fp.read()
        return Response(
            src,
            mimetype="text/csv",
            headers={"Content-disposition":
                     "attachment; filename=sample.txt"})
    
# ----- API -----
class UploadAPI(Resource):
    def get(self):
        print("check passed")
        return {"message": "API For TextExtractor2.0"}, 200
    
    def post(self):
        data = parser.parse_args()
        if data['file'] == "":
            return {'message':'No file found'}, 400
        
        photo = data['file']
        if photo:
            photo.save(os.path.join("./static/images/",photo.filename))
            img = Image.open(photo)
            img1 = img.convert("LA")
            text = pytesseract.image_to_string(img1)
            print("check 1 passed")
            os.remove(os.path.join("./static/images/",photo.filename))
            return {"message": text}, 200
        else:
            return {'message':'Something went wrong'}, 500

api.add_resource(UploadAPI, "/api/v1/")

# End Of API Endpoint
        
if __name__ == "__main__": 
        app.run()


    
    
