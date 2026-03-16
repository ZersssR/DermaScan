from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import os
import uuid
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'skindisease'
mysql = MySQL(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50()
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 6))
model.load_state_dict(torch.load("DermaScan/best_resnet50_skin_2.pth", map_location=device))
model.to(device)
model.eval()

classes = ["Eczema", "Melanoma", "Atopic Dermatitis", "Basal Cell Carcinoma (BCC)", "Melanocytic Nevi (NV)", "Benign Keratosis-like Lesions (BKL)"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_idx = torch.max(prob, 1)
        label = classes[pred_idx.item()]
        if confidence.item() < 0.70:
            label += " (Low Confidence)"
        return label, confidence.item()
    except Exception:
        return "Invalid image", 0.0

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        password = request.form['password']

        if uname == "Admin" and password == "Admin12":
            session['admin_id'] = 1
            session['admin_username'] = 'Admin'
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_panel'))

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM User WHERE Username = %s AND Password = %s", (uname, password))
        user = cur.fetchone()
        cur.close()

        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')

    return render_template('user/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        try:
            cur.execute("INSERT INTO user (Username, Email, Password) VALUES (%s, %s, %s)",
                        (username, email, password))
            mysql.connection.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            mysql.connection.rollback()
            flash(f'Registration failed: {str(e)}', 'error')
        finally:
            cur.close()
    return render_template('user/register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    states = get_states()
    clinics = None
    selected_state = None
    state_name = ""
    prediction = None
    history = []

    # Process POST request
    if request.method == 'POST':
        if 'state' in request.form:
            selected_state = request.form['state']
            clinics = get_clinics_by_state(selected_state)
            cur = mysql.connection.cursor()
            cur.execute("SELECT DISTINCT state_name FROM clinics WHERE state_id = %s", (selected_state,))
            result = cur.fetchone()
            if result:
                state_name = result[0]
            cur.close()

        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                ext = os.path.splitext(file.filename)[1]
                filename = f"{uuid.uuid4().hex}{ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                label, confidence = predict_image(filepath)
                confidence_percent = round(confidence * 100, 2)
                if confidence < 0.7:
                    label = "Not a skin disease"

                prediction = {
                    'disease': label,
                    'confidence': confidence_percent,
                    'image_url': f"/static/uploads/{filename}"
                }
                session['prediction'] = prediction

                if 'user_id' in session:
                    cur = mysql.connection.cursor()
                    cur.execute(
                        "INSERT INTO Results (UserID, Disease_name, Confidence, Image_name, TimeStamp) VALUES (%s, %s, %s, %s, %s)",
                        (session['user_id'], label, confidence_percent, filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    )
                    mysql.connection.commit()
                    cur.close()

    # Load history if logged in
    if 'user_id' in session:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        cur.execute("""SELECT TimeStamp, Disease_name, Confidence, Image_name 
                       FROM Results WHERE UserID = %s ORDER BY TimeStamp DESC""", (user_id,))
        results = cur.fetchall()
        cur.close()
        history = [{
            'date_time': r[0], 'disease': r[1], 'confidence': r[2], 'image_filename': r[3]
        } for r in results]

    return render_template('user/dashboard.html',
                           prediction=session.get('prediction'),
                           history=history,
                           states=states,
                           clinics=clinics,
                           selected_state=selected_state,
                           state_name=state_name)

def get_states():
    cur = mysql.connection.cursor()
    cur.execute("SELECT DISTINCT state_id, state_name FROM clinics")
    states = cur.fetchall()
    cur.close()
    return states

def get_clinics_by_state(state_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT clinic_name, address, phone, email FROM clinics WHERE state_id = %s", (state_id,))
    clinics = cur.fetchall()
    cur.close()
    return clinics

@app.route('/clinics', methods=['GET', 'POST'])
def clinics():
    states = get_states()
    clinics = None
    selected_state = None
    state_name = ""

    if request.method == 'POST':
        selected_state = request.form['state']
        clinics = get_clinics_by_state(selected_state)
        cur = mysql.connection.cursor()
        cur.execute("SELECT DISTINCT state_name FROM clinics WHERE state_id = %s", (selected_state,))
        result = cur.fetchone()
        if result:
            state_name = result[0]
        cur.close()

    return render_template('user/clinics.html', states=states, clinics=clinics, selected_state=selected_state, state_name=state_name)
@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Login required to view prediction history.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    cur = mysql.connection.cursor()
    cur.execute("SELECT TimeStamp, Disease_name, Confidence, Image_name FROM Results WHERE UserID = %s ORDER BY TimeStamp DESC", (user_id,))
    results = cur.fetchall()
    cur.close()

    prediction_history = [
        {
            'date_time': row[0],
            'disease': row[1],
            'confidence': row[2],
            'image_filename': row[3]
        } for row in results
    ]
    return render_template('user/history.html', prediction_history=prediction_history)

@app.route('/result')
def result():
    if 'user_id' not in session or 'prediction' not in session:
        return redirect(url_for('dashboard'))
    return render_template('user/result.html', prediction=session['prediction'])

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        new_username = request.form['username']
        new_email = request.form['email']
        new_password = request.form['password']

        try:
            if new_password.strip() != '':
                cur.execute("UPDATE User SET Username = %s, Email = %s, Password = %s WHERE UserID = %s",
                            (new_username, new_email, new_password, user_id))
            else:
                cur.execute("UPDATE User SET Username = %s, Email = %s WHERE UserID = %s",
                            (new_username, new_email, user_id))
            mysql.connection.commit()
            session['username'] = new_username
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            mysql.connection.rollback()
            flash(f'Update failed: {str(e)}', 'error')
        finally:
            cur.close()

    cur.execute("SELECT Username, Email FROM User WHERE UserID = %s", (user_id,))
    user_data = cur.fetchone()
    cur.close()

    if not user_data:
        flash('User not found', 'error')
        return redirect(url_for('dashboard'))

    user = {
        'username': user_data[0],
        'email': user_data[1]
    }
    return render_template('user/edit_profile.html', user=user)

@app.route('/disease/<name>')
def disease_info(name):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Disease_Treatment_info WHERE Disease_name = %s", (name,))
    row = cur.fetchone()
    cur.close()

    if not row:
        flash('Disease information not found', 'error')
        return redirect(url_for('dashboard'))

    data = {
        'id': row[0],
        'name': row[1],
        'info': row[2],
        'treatment': row[3]
    }

    return render_template('user/disease_info.html', data=data)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    user_id = request.form.get('user_id')
    disease = request.form.get('disease')
    confidence = request.form.get('confidence')
    image_name = request.form.get('image_name')
    helpful = request.form.get('helpful')
    reason = request.form.get('reason') if helpful == 'no' else ''

    cur = mysql.connection.cursor()
    try:
        cur.execute("""
            UPDATE Results
            SET Helpful = %s, Reason = %s
            WHERE UserID = %s AND Image_name = %s
        """, (helpful, reason, user_id, image_name))
        mysql.connection.commit()
        flash('Feedback submitted successfully.', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Error submitting feedback: {str(e)}', 'error')
    finally:
        cur.close()

    return redirect(url_for('dashboard'))

# ========== ADMIN ROUTES ==========

@app.route('/admin')
def admin_panel():
    if 'admin_id' not in session:
        flash('Login required.', 'error')
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    # Fetch chart data first
    cur.execute("SELECT Disease_name, COUNT(*) as count FROM Results GROUP BY Disease_name")
    disease_stats = cur.fetchall()

    # Then fetch stats for cards
    cur.execute("SELECT COUNT(*) FROM User")
    total_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM Disease_Treatment_info")
    total_diseases = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM Clinics")
    total_clinics = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM Results")
    total_predictions = cur.fetchone()[0]

    cur.close()

    return render_template(
        'admin/admin_dashboard.html',
        disease_stats=disease_stats,
        total_users=total_users,
        total_diseases=total_diseases,
        total_clinics=total_clinics,
        total_predictions=total_predictions
    )

@app.route('/admin/clinics')
def manage_clinics():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM States ORDER BY state_name")
    states = cur.fetchall()

    cur.execute("""
        SELECT c.*, s.state_name 
        FROM Clinics c
        JOIN States s ON c.state_id = s.state_id
        ORDER BY s.state_name, c.clinic_name
    """)
    clinics = cur.fetchall()
    cur.close()

    return render_template('admin/admin_clinics.html', states=states, clinics=clinics)


@app.route('/admin/add_clinic', methods=['POST'])
def add_clinic():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    clinic_name = request.form['clinic_name']
    state_id = request.form['state_id']
    address = request.form['address']
    phone = request.form.get('phone', '')
    email = request.form.get('email', '')

    cur = mysql.connection.cursor()
    try:
        cur.execute("""
            INSERT INTO Clinics (state_id, clinic_name, address, phone, email)
            VALUES (%s, %s, %s, %s, %s)
        """, (state_id, clinic_name, address, phone, email))
        mysql.connection.commit()
        flash('Clinic added successfully!', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Error: {str(e)}', 'error')
    finally:
        cur.close()

    return redirect(url_for('manage_clinics'))


@app.route('/admin/delete_clinic/<int:clinic_id>')
def delete_clinic(clinic_id):
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    try:
        cur.execute("DELETE FROM Clinics WHERE clinic_id = %s", (clinic_id,))
        mysql.connection.commit()
        flash('Clinic deleted.', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Error: {str(e)}', 'error')
    finally:
        cur.close()

    return redirect(url_for('manage_clinics'))

@app.route('/admin/update_clinic', methods=['POST'])
def update_clinic():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    clinic_id = request.form['clinic_id']
    clinic_name = request.form['clinic_name']
    state_id = request.form['state_id']
    address = request.form['address']
    phone = request.form.get('phone', '')
    email = request.form.get('email', '')

    cur = mysql.connection.cursor()
    try:
        cur.execute("""
            UPDATE Clinics
            SET state_id=%s, clinic_name=%s, address=%s, phone=%s, email=%s
            WHERE clinic_id=%s
        """, (state_id, clinic_name, address, phone, email, clinic_id))
        mysql.connection.commit()
        flash("Clinic updated successfully!", "success")
    except Exception as e:
        mysql.connection.rollback()
        flash(f"Error updating clinic: {str(e)}", "error")
    finally:
        cur.close()

    return redirect(url_for('manage_clinics'))

@app.route('/admin/lowconfidence')
def low_confidence_monitor():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT R.Disease_name, R.Confidence, R.Image_name, R.TimeStamp,
               U.Username, R.Helpful, R.Reason
        FROM Results R
        JOIN User U ON R.UserID = U.UserID
        ORDER BY R.TimeStamp DESC
    """)
    raw = cur.fetchall()
    cur.close()

    low_conf = []
    for row in raw:
        low_conf.append({
            'disease': row[0],
            'confidence': row[1],
            'image': row[2],
            'timestamp': row[3],
            'username': row[4],
            'helpful': row[5],
            'reason': row[6]
        })

    return render_template('admin/admin_lowconfidence.html', low_conf=low_conf)

@app.route('/admin/disease')
def admin_disease():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Disease_Treatment_info")
    diseases = cur.fetchall()
    cur.close()

    return render_template('admin/admin_disease.html', diseases=diseases)


@app.route('/admin/add_disease', methods=['POST'])
def add_disease():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    name = request.form['name']
    info = request.form['info']
    treatment = request.form['treatment']

    cur = mysql.connection.cursor()
    try:
        cur.execute("""
            INSERT INTO Disease_Treatment_info (Disease_name, Disease_info, Treatment_info)
            VALUES (%s, %s, %s)
        """, (name, info, treatment))
        mysql.connection.commit()
        flash('Disease added!', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Error: {str(e)}', 'error')
    finally:
        cur.close()

    return redirect(url_for('admin_disease'))


@app.route('/admin/update_disease/<int:id>', methods=['POST'])
def update_disease(id):
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    name = request.form['name']
    info = request.form['info']
    treatment = request.form['treatment']

    cur = mysql.connection.cursor()
    try:
        cur.execute("""
            UPDATE Disease_Treatment_info
            SET Disease_name = %s, Disease_info = %s, Treatment_info = %s
            WHERE Disease_ID = %s
        """, (name, info, treatment, id))
        mysql.connection.commit()
        flash('Disease updated!', 'success')
    except Exception as e:
        mysql.connection.rollback()
        flash(f'Error: {str(e)}', 'error')
    finally:
        cur.close()

    return redirect(url_for('admin_disease'))

@app.route('/admin/delete_disease/<int:id>')
def delete_disease(id):
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    try:
        cur.execute("DELETE FROM Disease_Treatment_info WHERE Disease_ID = %s", (id,))
        mysql.connection.commit()
        flash("Disease deleted successfully!", "success")
    except Exception as e:
        mysql.connection.rollback()
        flash(f"Error deleting disease: {str(e)}", "error")
    finally:
        cur.close()

    return redirect(url_for('admin_disease'))

@app.route('/admin/users')
def admin_users():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM User")
    users = cur.fetchall()
    cur.close()

    return render_template('admin/admin_users.html', users=users)


@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_id', None)
    session.pop('admin_username', None)
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
