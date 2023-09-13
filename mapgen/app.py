import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

from tensorflow.keras.models import load_model  

app = Flask(__name__)

modelo_gan = load_model('ganmap.keras') 

@app.route('/generate_image', methods=['GET'])
def generate_image():
    try:
        data = request.get_json()

        generated_image = modelo_gan.generate_image(noise)

        image_bytes = generated_image.tobytes()

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        response_data = {'image': image_base64}

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    print(f'Server running at http://{host}:{port}')
    app.run(host=host, port=port)
