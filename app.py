from flask import Flask, render_template, request, redirect, url_for, session
import os, time, datetime, sqlite3
import base64, json, random
from datetime import datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io
from Courses import ds_course, web_course, android_course, ios_course, uiux_course
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from plotly.io import to_html



# === Configurations ===
UPLOAD_FOLDER = 'Uploaded_Resumes'
genai.configure(api_key="AIzaSyCwbDCula-Gg7UecSGj4Jp8wnEPuU8Pg4M")
model = genai.GenerativeModel("gemini-1.5-flash")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Secret key for session management

# === DB ===
conn = sqlite3.connect('sra_database.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS user_data (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL, Email_ID TEXT NOT NULL, resume_score TEXT NOT NULL, Timestamp TEXT NOT NULL,
    Page_no TEXT NOT NULL, Predicted_Field TEXT NOT NULL, User_level TEXT NOT NULL,
    Actual_skills TEXT NOT NULL, Recommended_skills TEXT NOT NULL, Recommended_courses TEXT NOT NULL
)''')
conn.commit()

# === Helpers ===
def pdf_reader(file_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    with open(file_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            interpreter = PDFPageInterpreter(resource_manager, converter)
            interpreter.process_page(page)
    text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def calculate_ats_score(job_desc, resume_text):
    job_embedding = bert_model.encode(job_desc, convert_to_tensor=True)
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)
    return round(float(np.dot(job_embedding, resume_embedding) / (np.linalg.norm(job_embedding) * np.linalg.norm(resume_embedding))) * 100, 2)

def course_recommender(course_list, count=4):
    random.shuffle(course_list)
    return course_list[:count]

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    cursor.execute("""
        INSERT INTO user_data (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, str(skills), str(recommended_skills), str(courses)))
    conn.commit()

# === Routes ===

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get file and job description from the form
        file = request.files['resume']
        job_desc = request.form.get('job_description')

        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Extract resume data using ResumeParser
            resume_data = ResumeParser(file_path).get_extracted_data()
            resume_text = pdf_reader(file_path)

            name = resume_data.get('name', 'Unknown')
            email = resume_data.get('email', '')
            pages = resume_data.get('no_of_pages', 1)
            skills = resume_data.get('skills', [])

            # Determine user level based on resume pages
            if pages == 1:
                level = "Fresher"
            elif pages == 2:
                level = "Intermediate"
            else:
                level = "Experienced"

            # Predict the field and recommend courses based on the skills
            reco_field = ""
            recommended_skills = []
            recommended_courses = []

            fields = {
                'Data Science': ds_course,
                'Web Development': web_course,
                'Android Development': android_course,
                'IOS Development': ios_course,
                'UI-UX Development': uiux_course,
            }

            keywords = {
                'Data Science': ['tensorflow', 'keras', 'pytorch'],
                'Web Development': ['react', 'django', 'node js'],
                'Android Development': ['android', 'kotlin', 'flutter'],
                'IOS Development': ['swift', 'xcode', 'ios'],
                'UI-UX Development': ['figma', 'adobe xd', 'ux', 'ui'],
            }

            # Identify the field based on the skills in the resume
            for field, kws in keywords.items():
                if any(skill.lower() in kws for skill in skills):
                    reco_field = field
                    recommended_courses = course_recommender(fields[field])
                    recommended_skills = kws
                    break

            # Calculate the timestamp for when the resume was processed
            ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

            # Calculate the resume score based on specific keywords
            resume_score = 0
            for keyword in ['Objective', 'Declaration', 'Hobbies']:
                if keyword.lower() in resume_text.lower():
                    resume_score += 20
            if len(skills) > 3:
                resume_score += 20

            # Insert the extracted data into the database
            insert_data(name, email, resume_score, ts, pages, reco_field, level, skills, recommended_skills, recommended_courses)

            # Calculate ATS score and provide feedback if job description is given
            ats_score = 0
            ats_feedback = ""
            if job_desc:
                ats_score = calculate_ats_score(job_desc, resume_text)
                gemini_prompt = f"""
                    You are an ATS (Applicant Tracking System) scanner with expertise in evaluating resumes against job descriptions.
                    Your task is to assess the resume's fit against the provided job description. Please output the match percentage and missing keywords, followed by final thoughts.

                    Resume:
                    {resume_text}

                    Job Description:
                    {job_desc}

                    Provide your suggestions in clear bullet points and be concise.
                """
                response = model.generate_content(gemini_prompt)
                ats_feedback = response.text

            # Fetch user data for admin view (to display user data and charts)
            cursor.execute('SELECT * FROM user_data')
            data = cursor.fetchall()

            # Prepare the records for display on the admin dashboard
            records = [
                {
                    'name': row[1],
                    'email': row[2],
                    'resume_score': row[3],
                    'timestamp': row[4],
                    'predicted_field': row[5],
                    'user_level': row[6],
                    'recommended_skills': row[7],
                    'recommended_courses': row[8]
                }
                for row in data
            ]

            # Prepare data for pie charts
            plot_data = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page', 'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills', 'Recommended Course'])
            predicted_field_data = plot_data['Predicted Field'].value_counts()
            user_level_data = plot_data['User Level'].value_counts()

            return render_template(
                'result.html',
                records=records,
                predicted_field_data=predicted_field_data.to_dict(),
                user_level_data=user_level_data.to_dict(),
                name=name,
                email=email,
                level=level,
                reco_field=reco_field,
                skills=skills,
                ats_score=ats_score,
                ats_feedback=ats_feedback,
                resume_score=resume_score,
                recommended_courses=recommended_courses,
                current_year=datetime.now().year
            )

    return render_template('index.html', current_year="2025")
import csv
from flask import send_file

@app.route('/download_csv')
def download_csv():
    # Prepare data (e.g., fetch data from the database)
    cursor.execute('SELECT * FROM user_data')
    data = cursor.fetchall()

    # Define the filename for the CSV
    filename = "user_data.csv"

    # Create CSV in memory
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers to CSV
        writer.writerow(['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page', 'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills', 'Recommended Course'])
        # Write the rows to the CSV
        for row in data:
            writer.writerow(row)

    # Return the CSV as a downloadable file
    return send_file(filename, as_attachment=True)

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error="Invalid credentials! Try again.")

    return render_template('admin_login.html')

# Sample database connection function
def get_db_connection():
    conn = sqlite3.connect('sra_database.db')  # Update with your actual database name
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    return conn
@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get search query and pagination values
    search_query = request.args.get('search', '')
    page = int(request.args.get('page', 1))  # Default to page 1
    per_page = 10  # Number of records per page
    offset = (page - 1) * per_page  # Calculate the offset for pagination

    # SQL query to fetch records based on search query with pagination
    if search_query:
        query = "SELECT * FROM user_data WHERE Name LIKE ? OR Email_ID LIKE ? LIMIT ? OFFSET ?"
        cursor.execute(query, (f'%{search_query}%', f'%{search_query}%', per_page, offset))
    else:
        cursor.execute("SELECT * FROM user_data LIMIT ? OFFSET ?", (per_page, offset))

    records = cursor.fetchall()

    # Get the total number of records for pagination
    cursor.execute("SELECT COUNT(*) FROM user_data")
    total_records = cursor.fetchone()[0]

    # Calculate total pages
    total_pages = (total_records // per_page) + (1 if total_records % per_page > 0 else 0)

    # Prepare data for the charts
    predicted_field_data = Counter(row['Predicted_Field'] for row in records)
    user_level_data = Counter(row['User_Level'] for row in records)

    # Prepare pie chart for Predicted Field distribution
    predicted_field_df = pd.DataFrame(predicted_field_data.items(), columns=['Field', 'Count'])
    fig1 = px.pie(predicted_field_df, values='Count', names='Field', title="Predicted Field Distribution")
    predicted_field_chart = to_html(fig1, full_html=False)

    # Prepare pie chart for User Level distribution
    user_level_df = pd.DataFrame(user_level_data.items(), columns=['Level', 'Count'])
    fig2 = px.pie(user_level_df, values='Count', names='Level', title="User Level Distribution")
    user_level_chart = to_html(fig2, full_html=False)

    # Close the connection
    conn.close()

    # Passing data to the template
    return render_template('admin.html',
                           records=records,
                           predicted_field_data=predicted_field_data,
                           user_level_data=user_level_data,
                           predicted_field_chart=predicted_field_chart,
                           user_level_chart=user_level_chart,
                           page=page,
                           total_pages=total_pages,
                           search_query=search_query,
                           current_year=2025)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# === Run ===
if __name__ == '__main__':
    app.run("0.0.0.0",5000)
