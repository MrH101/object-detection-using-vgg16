from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import model
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
Bootstrap(app)

ALLOWED_EXTENSIONS = set(['avi', 'mp4', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    uploaded = False
    return render_template('index.html', uploaded = uploaded)

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    analysed = False
    loading = True
    uploadDirectory = "uploads/"
    if request.method == 'POST':
        f = request.files['file']
        f.save("uploads/video.mp4")
        analysed = model.splitVideo('uploads/video.mp4')
        if(analysed):
        	loading = False
        	return render_template('index.html', analysis_done = analysed, loading = loading)


@app.route('/search', methods = ['POST'])
def search_df():
      query = request.form['search']
      processed_text = query.upper()
      print(query)
      images = model.search(query=query)
      return render_template('index.html', uploaded = True, loading = False, images = images, analysis_done = True, showImages = True)


if __name__ == '__main__':
    app.run()
