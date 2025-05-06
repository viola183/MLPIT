import pandas as pd
import numpy as np
import re
import pickle
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class SpamDetectorModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.features = None
        
    def preprocess_data(self, df):
        """Preprocess the email data"""
        # Create new features from text content
        df['content_length'] = df['content'].apply(len)
        df['has_url'] = df['content'].apply(lambda x: 1 if 'http' in x.lower() or '.com' in x.lower() or '.net' in x.lower() or '.org' in x.lower() or '.info' in x.lower() or '.biz' in x.lower() else 0)
        df['has_money'] = df['content'].apply(lambda x: 1 if '$' in x or 'money' in x.lower() or 'cash' in x.lower() or 'offer' in x.lower() else 0)
        df['has_urgent'] = df['content'].apply(lambda x: 1 if 'urgent' in x.lower() or 'alert' in x.lower() or 'attention' in x.lower() or 'important' in x.lower() else 0)
        
        # Extract features from content
        df['exclamation_count'] = df['content'].apply(lambda x: x.count('!'))
        df['question_count'] = df['content'].apply(lambda x: x.count('?'))
        df['caps_ratio'] = df['content'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        
        return df
    
    def extract_features(self, df):
        """Extract features for model training"""
        # Select relevant features
        features = ['content_length', 'spam_indicators', 'has_url', 'has_money', 'has_urgent', 
                   'exclamation_count', 'question_count', 'caps_ratio']
        X = df[features]
        y = df['is_spam']
        
        self.features = features
        return X, y
    
    def train(self, csv_path):
        """Train the decision tree model"""
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        df = self.preprocess_data(df)
        
        # Extract features and target
        X, y = self.extract_features(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train the model
        self.model = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report
    
    def predict(self, email_content, content_length=None, spam_indicators=0):
        """Predict if an email is spam based on its content"""
        if content_length is None:
            content_length = len(email_content)
        
        # Extract features from the email content
        has_url = 1 if 'http' in email_content.lower() or '.com' in email_content.lower() or '.net' in email_content.lower() or '.org' in email_content.lower() or '.info' in email_content.lower() or '.biz' in email_content.lower() else 0
        has_money = 1 if '$' in email_content or 'money' in email_content.lower() or 'cash' in email_content.lower() or 'offer' in email_content.lower() else 0
        has_urgent = 1 if 'urgent' in email_content.lower() or 'alert' in email_content.lower() or 'attention' in email_content.lower() or 'important' in email_content.lower() else 0
        exclamation_count = email_content.count('!')
        question_count = email_content.count('?')
        caps_ratio = sum(1 for c in email_content if c.isupper()) / len(email_content) if len(email_content) > 0 else 0
        
        # Create a dataframe with a single row containing the features
        features_df = pd.DataFrame({
            'content_length': [content_length],
            'spam_indicators': [spam_indicators],
            'has_url': [has_url],
            'has_money': [has_money],
            'has_urgent': [has_urgent],
            'exclamation_count': [exclamation_count],
            'question_count': [question_count],
            'caps_ratio': [caps_ratio]
        })
        
        # Make prediction
        prediction = self.model.predict(features_df)[0]
        probability = self.model.predict_proba(features_df)[0][1]  # Probability of being spam
        
        return prediction, probability
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        model_data = {
            'model': self.model,
            'features': self.features
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model from a file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.features = model_data['features']


class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Spam Detector")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        self.detector = SpamDetectorModel()
        self.setup_ui()
        
        # Try to load a pre-trained model if available
        try:
            self.detector.load_model("spam_detector_model.pkl")
            self.status_var.set("Model loaded successfully!")
        except:
            self.status_var.set("No pre-trained model found. Please train the model first.")
    
    def setup_ui(self):
        """Set up the user interface"""
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create tabs
        self.tab_predict = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_about = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_predict, text="Predict")
        self.notebook.add(self.tab_train, text="Train Model")
        self.notebook.add(self.tab_about, text="About")
        
        # Set up each tab
        self.setup_predict_tab()
        self.setup_train_tab()
        self.setup_about_tab()
        
        # Status bar at the bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_predict_tab(self):
        """Set up the prediction tab"""
        # Email input frame
        input_frame = ttk.LabelFrame(self.tab_predict, text="Email Content")
        input_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Create sender, subject and content fields
        ttk.Label(input_frame, text="From:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.sender_entry = ttk.Entry(input_frame, width=50)
        self.sender_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(input_frame, text="Subject:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.subject_entry = ttk.Entry(input_frame, width=50)
        self.subject_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(input_frame, text="Content:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W+tk.N)
        self.content_text = scrolledtext.ScrolledText(input_frame, width=50, height=10, wrap=tk.WORD)
        self.content_text.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E+tk.N+tk.S)
        
        # Advanced features
        advanced_frame = ttk.LabelFrame(self.tab_predict, text="Advanced Features (Optional)")
        advanced_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Label(advanced_frame, text="Spam Indicators:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.spam_indicators_var = tk.IntVar(value=0)
        self.spam_indicators_entry = ttk.Spinbox(advanced_frame, from_=0, to=10, textvariable=self.spam_indicators_var, width=5)
        self.spam_indicators_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.tab_predict, text="Prediction Results")
        results_frame.pack(padx=10, pady=10, fill="x")
        
        self.result_var = tk.StringVar()
        self.result_var.set("No prediction yet")
        self.result_label = ttk.Label(results_frame, textvariable=self.result_var, font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)
        
        self.probability_var = tk.StringVar()
        self.probability_label = ttk.Label(results_frame, textvariable=self.probability_var)
        self.probability_label.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.tab_predict)
        button_frame.pack(padx=10, pady=10, fill="x")
        
        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.predict_email)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_predict_form)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Sample emails
        sample_frame = ttk.LabelFrame(self.tab_predict, text="Sample Emails")
        sample_frame.pack(padx=10, pady=10, fill="x")
        
        self.sample_regular_button = ttk.Button(sample_frame, text="Load Regular Email Example", 
                                               command=lambda: self.load_sample_email("regular"))
        self.sample_regular_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.sample_spam_button = ttk.Button(sample_frame, text="Load Spam Email Example", 
                                            command=lambda: self.load_sample_email("spam"))
        self.sample_spam_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    def setup_train_tab(self):
        """Set up the training tab"""
        # Dataset input frame
        input_frame = ttk.LabelFrame(self.tab_train, text="Dataset")
        input_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Label(input_frame, text="CSV File Path:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.csv_path_var = tk.StringVar(value="email_spam_dataset.csv")
        self.csv_path_entry = ttk.Entry(input_frame, textvariable=self.csv_path_var, width=50)
        self.csv_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Training button
        button_frame = ttk.Frame(self.tab_train)
        button_frame.pack(padx=10, pady=10, fill="x")
        
        self.train_button = ttk.Button(button_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        # Training results frame
        results_frame = ttk.LabelFrame(self.tab_train, text="Training Results")
        results_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.train_result_text = scrolledtext.ScrolledText(results_frame, width=70, height=15, wrap=tk.WORD)
        self.train_result_text.pack(padx=5, pady=5, fill="both", expand=True)
    
    def setup_about_tab(self):
        """Set up the about tab"""
        about_frame = ttk.Frame(self.tab_about)
        about_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        about_text = """
        Email Spam Detector
        
        This application uses a Decision Tree model to classify emails as spam or legitimate.
        
        Features used for classification:
        - Content length
        - Presence of URLs or domain names
        - References to money or offers
        - Urgency indicators
        - Punctuation patterns
        - Use of capital letters
        - Other spam indicators
        
        Instructions:
        1. Train the model using the "Train Model" tab (or use the pre-loaded model).
        2. Enter email details in the "Predict" tab.
        3. Click "Predict" to classify the email.
        
        For best results, provide complete email content with headers.
        """
        
        about_label = ttk.Label(about_frame, text=about_text, wraplength=600, justify=tk.LEFT)
        about_label.pack(padx=20, pady=20)
    
    def predict_email(self):
        """Predict if an email is spam"""
        if self.detector.model is None:
            messagebox.showerror("Error", "No model has been trained or loaded. Please train a model first.")
            return
        
        # Get input data
        sender = self.sender_entry.get()
        subject = self.subject_entry.get()
        content = self.content_text.get("1.0", tk.END).strip()
        spam_indicators = self.spam_indicators_var.get()
        
        if not content:
            messagebox.showerror("Error", "Please enter email content for prediction.")
            return
        
        # Combine email parts for better prediction
        full_text = f"From: {sender}\nSubject: {subject}\n\n{content}"
        
        # Make prediction
        prediction, probability = self.detector.predict(full_text, len(full_text), spam_indicators)
        
        # Update result
        if prediction == 1:
            self.result_var.set("SPAM")
            self.result_label.config(foreground="red")
        else:
            self.result_var.set("NOT SPAM")
            self.result_label.config(foreground="green")
        
        self.probability_var.set(f"Confidence: {probability*100:.2f}%")
        self.status_var.set("Prediction complete")
    
    def clear_predict_form(self):
        """Clear the prediction form"""
        self.sender_entry.delete(0, tk.END)
        self.subject_entry.delete(0, tk.END)
        self.content_text.delete("1.0", tk.END)
        self.spam_indicators_var.set(0)
        self.result_var.set("No prediction yet")
        self.probability_var.set("")
        self.result_label.config(foreground="black")
        self.status_var.set("Form cleared")
    
    def load_sample_email(self, email_type):
        """Load a sample email for testing"""
        self.clear_predict_form()
        
        if email_type == "regular":
            self.sender_entry.insert(0, "john.smith@company.com")
            self.subject_entry.insert(0, "Meeting tomorrow")
            sample_content = """Hi Team,

Just a reminder about our project status meeting tomorrow at 10:00 AM in Conference Room B. Please prepare a brief update on your assigned tasks.

Looking forward to seeing everyone there.

Best regards,
John"""
            self.content_text.insert("1.0", sample_content)
            self.spam_indicators_var.set(0)
            
        elif email_type == "spam":
            self.sender_entry.insert(0, "prize-notify@win-big.com")
            self.subject_entry.insert(0, "CONGRATULATIONS! YOU'VE WON!!!")
            sample_content = """CONGRATULATIONS!!!

YOU'VE WON THE INTERNATIONAL LOTTERY WORTH $5,000,000.00 USD!

To claim your prize money, please contact our legal team immediately:
Email: claims@secure-transfer.net
Ref: WIN-XB5721

THIS IS URGENT! You must respond within 24 HOURS or forfeit your prize!

Limited time opportunity - ACT NOW!!!"""
            self.content_text.insert("1.0", sample_content)
            self.spam_indicators_var.set(3)
        
        self.status_var.set(f"Loaded {email_type} email example")
    
    def train_model(self):
        """Train the spam detection model"""
        csv_path = self.csv_path_var.get()
        
        if not csv_path:
            messagebox.showerror("Error", "Please enter a valid CSV file path.")
            return
        
        self.status_var.set("Training model... Please wait.")
        self.root.update()
        
        try:
            # Train the model
            accuracy, report = self.detector.train(csv_path)
            
            # Save the model
            self.detector.save_model("spam_detector_model.pkl")
            
            # Display results
            result_text = f"Model trained successfully!\n\n"
            result_text += f"Accuracy: {accuracy:.4f}\n\n"
            result_text += "Classification Report:\n"
            result_text += report
            
            self.train_result_text.delete("1.0", tk.END)
            self.train_result_text.insert("1.0", result_text)
            
            self.status_var.set("Model trained and saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
            self.status_var.set("Error training model.")


def main():
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()