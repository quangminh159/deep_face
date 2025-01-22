from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from compare_face import detect_and_crop_face, compare_images
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'image1' not in request.files or 'image2' not in request.files:
        return redirect(request.url)

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        return redirect(request.url)

    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        image_path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        image_path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        face1, _ = detect_and_crop_face(image_path1)
        face2, _ = detect_and_crop_face(image_path2)

        if face1 is None or face2 is None:
            return render_template('result.html', error="Không phát hiện được khuôn mặt trong một hoặc cả hai hình ảnh.")
        
        cropped_path1 = os.path.join(app.config['UPLOAD_FOLDER'], "cropped_" + filename1)
        cropped_path2 = os.path.join(app.config['UPLOAD_FOLDER'], "cropped_" + filename2)
        cv2.imwrite(cropped_path1, face1)
        cv2.imwrite(cropped_path2, face2)
        similarity, is_similar = compare_images(cropped_path1, cropped_path2)

        return render_template('result.html', similarity=similarity, is_similar=is_similar,
                               image1="cropped_" + filename1, image2="cropped_" + filename2)

    return redirect(request.url)
upload_folder = os.path.join(app.static_folder, 'uploads')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
app.config['UPLOAD_FOLDER'] = upload_folder

if __name__ == '__main__':
    app.run(debug=True)
