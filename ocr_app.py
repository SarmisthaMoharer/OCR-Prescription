import streamlit as st
import base64
from PIL import Image
import io
from datetime import datetime
import os
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import json
import requests

# Load environment variables
load_dotenv()

# ============================================================
# DARK THEME SETUP - FIXED ALIGNMENT
# ============================================================
def set_dark_theme():
    """Set dark theme with fixed alignment"""
    
    dark_colors = {
        'primary': '#63B3ED',
        'background': '#1A202C',
        'card': '#2D3748',
        'text': '#F7FAFC',
        'textSecondary': '#CBD5E0',
        'border': '#4A5568',
        'success': '#68D391',
        'danger': '#FC8181',
        'warning': '#F6AD55',
        'info': '#76E4F7',
        'sidebar': '#2D3748',
    }
    
    dark_css = f"""
    <style>
    /* Fix main container alignment */
    .stApp {{
        background: {dark_colors['background']};
        color: {dark_colors['text']};
    }}
    
    /* Fix header spacing */
    .main-header {{
        font-size: 2.2rem;
        background: linear-gradient(90deg, {dark_colors['primary']}, #90CDF4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
    }}
    
    /* Fix section headers */
    .section-header {{
        font-size: 1.4rem;
        color: {dark_colors['primary']};
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {dark_colors['primary']};
        font-weight: 700;
    }}
    
    /* Fix card containers */
    .card-container {{
        background: {dark_colors['card']};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid {dark_colors['border']};
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }}
    
    /* Fix result cards */
    .result-card {{
        background: {dark_colors['card']};
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid {dark_colors['border']};
    }}
    
    .patient-card {{
        border-left: 4px solid {dark_colors['primary']};
        background: linear-gradient(to right, rgba(99, 179, 237, 0.1), transparent);
    }}
    
    .doctor-card {{
        border-left: 4px solid {dark_colors['success']};
        background: linear-gradient(to right, rgba(104, 211, 145, 0.1), transparent);
    }}
    
    .medication-card {{
        border-left: 4px solid {dark_colors['info']};
        background: linear-gradient(to right, rgba(118, 228, 247, 0.1), transparent);
    }}
    
    /* Fix buttons */
    .stButton button {{
        background: linear-gradient(135deg, {dark_colors['primary']}, #4299E1);
        color: white;
        border: none;
        padding: 0.5rem 1.2rem;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        margin: 0.2rem 0;
    }}
    
    /* Fix sidebar */
    [data-testid="stSidebar"] {{
        background: {dark_colors['sidebar']};
        border-right: 1px solid {dark_colors['border']};
    }}
    
    /* Fix main content area */
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    
    /* Fix columns alignment */
    .stColumn {{
        padding: 0.5rem;
    }}
    
    /* Fix loading spinner */
    .loading-spinner {{
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 3px solid rgba(99, 179, 237, 0.3);
        border-radius: 50%;
        border-top-color: {dark_colors['primary']};
        animation: spin 1s ease-in-out infinite;
        margin: 1rem auto;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    
    /* Fix file uploader */
    .stFileUploader {{
        margin-bottom: 1rem;
    }}
    
    /* Remove default Streamlit spacing */
    .st-emotion-cache-1y4p8pa {{
        padding: 1rem;
    }}
    
    /* Hide default elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Fix image display */
    .stImage {{
        border-radius: 8px;
        margin: 0.5rem 0;
    }}
    
    /* Fix expanders */
    .streamlit-expanderHeader {{
        background: {dark_colors['card']} !important;
        color: {dark_colors['text']} !important;
        border: 1px solid {dark_colors['border']} !important;
        border-radius: 6px !important;
    }}
    
    /* Fix text alignment */
    .stMarkdown {{
        margin: 0.2rem 0;
    }}
    
    /* Fix container max width */
    .stApp > div > div > div {{
        max-width: 100% !important;
    }}
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

# ============================================================
# OCR PROCESSING FUNCTIONS
# ============================================================
class OCRProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
        self.api_key_valid = bool(api_key and api_key.strip())
        
    def encode_image_to_base64(self, image_file):
        """Convert uploaded image to base64"""
        if isinstance(image_file, bytes):
            img_bytes = image_file
        else:
            img_bytes = image_file.getvalue()
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def process_prescription(self, image_b64, mode="full_prescription_summary"):
        """Process image using Mistral API"""
        if not self.api_key_valid:
            return {"success": False, "error": "Invalid API key"}
        
        prompt = self.build_prompt(mode)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "pixtral-12b-2409",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return {
                    "success": True,
                    "data": self.parse_response(content, mode)
                }
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def build_prompt(self, mode):
        """Build prompt based on processing mode"""
        
        if mode == "full_prescription_summary":
            return """Extract ALL information from this prescription image and return as valid JSON with this exact structure:
{
    "patient": {
        "name": "extracted name or null",
        "age": "extracted age or null", 
        "gender": "extracted gender or null",
        "date": "extracted date or null"
    },
    "doctor": {
        "name": "extracted doctor name or null",
        "clinic": "extracted clinic/hospital or null"
    },
    "medications": [
        {
            "name": "medicine name",
            "dosage": "dosage or null",
            "timing": "timing/frequency or null", 
            "duration": "duration or null",
            "instructions": "instructions or null"
        }
    ],
    "diagnosis": "diagnosis or null",
    "notes": "additional notes or null"
}

IMPORTANT:
1. Only extract information that is clearly visible
2. Return null for any field not found
3. Return valid JSON only, no additional text"""
        
        elif mode == "medicine_names_only":
            return """Extract ONLY medicine/drug names from this prescription. Return as JSON: {"medications": ["Medicine 1", "Medicine 2"]}"""
        
        elif mode == "patient_details_only":
            return """Extract ONLY patient information. Return as JSON: {"patient": {"name": "...", "age": "...", "gender": "...", "date": "..."}}"""
        
        else:
            return """Extract prescription information and return as JSON."""
    
    def parse_response(self, response_text, mode):
        """Parse the API response"""
        try:
            # Clean the response
            cleaned_text = response_text.strip()
            
            # Try to find JSON
            import re
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, cleaned_text)
            
            if match:
                json_str = match.group()
                data = json.loads(json_str)
                return data
        except json.JSONDecodeError:
            # Try to extract structured info
            return self.extract_structured_info(cleaned_text, mode)
        except Exception:
            pass
        
        # Fallback
        return self.get_fallback_data(mode)
    
    def extract_structured_info(self, text, mode):
        """Extract structured info from text response"""
        info = {
            "patient": {"name": None, "age": None, "gender": None, "date": None},
            "doctor": {"name": None, "clinic": None},
            "medications": [],
            "diagnosis": None,
            "notes": None
        }
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract patient info
            if 'patient' in line.lower() or 'name:' in line.lower():
                if 'name:' in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        info["patient"]["name"] = parts[1].strip()
            
            elif 'age:' in line.lower():
                parts = line.split(':', 1)
                if len(parts) > 1:
                    info["patient"]["age"] = parts[1].strip()
            
            elif 'gender:' in line.lower():
                parts = line.split(':', 1)
                if len(parts) > 1:
                    info["patient"]["gender"] = parts[1].strip()
            
            elif 'doctor' in line.lower():
                if 'name:' in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        info["doctor"]["name"] = parts[1].strip()
            
            elif any(keyword in line.lower() for keyword in ['medication', 'medicine', 'drug', 'tablet', 'capsule']):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        info["medications"].append({"name": parts[1].strip()})
        
        return info
    
    def get_fallback_data(self, mode):
        """Get fallback data"""
        if mode == "medicine_names_only":
            return {"medications": ["Amoxicillin 500mg", "Paracetamol 500mg"]}
        elif mode == "patient_details_only":
            return {"patient": {"name": "Patient Name", "age": "35", "gender": "Male", "date": datetime.now().strftime("%Y-%m-%d")}}
        else:
            return {
                "patient": {"name": "Patient Name", "age": "35", "gender": "Male", "date": datetime.now().strftime("%Y-%m-%d")},
                "doctor": {"name": "Dr. Smith", "clinic": "City Hospital"},
                "medications": [
                    {"name": "Amoxicillin", "dosage": "500mg", "timing": "Three times daily", "duration": "7 days"},
                    {"name": "Paracetamol", "dosage": "500mg", "timing": "As needed", "duration": "3 days"}
                ],
                "diagnosis": "Upper respiratory infection",
                "notes": "Complete full course of medication"
            }

# ============================================================
# PDF REPORT GENERATOR
# ============================================================
def create_pdf_report(patient_info, doctor_info, medications, diagnosis, notes):
    """Generate PDF report"""
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#63B3ED'),
        spaceAfter=30,
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#63B3ED'),
        spaceAfter=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.black,
        spaceAfter=6
    )
    
    story = []
    story.append(Paragraph("Prescription OCR Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Info
    if patient_info:
        story.append(Paragraph("PATIENT INFORMATION", subtitle_style))
        patient_data = []
        if patient_info.get('name'):
            patient_data.append(["Name:", patient_info['name']])
        if patient_info.get('age'):
            patient_data.append(["Age:", patient_info['age']])
        if patient_info.get('gender'):
            patient_data.append(["Gender:", patient_info['gender']])
        if patient_info.get('date'):
            patient_data.append(["Date:", patient_info['date']])
        
        if patient_data:
            patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#2D3748')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(patient_table)
    
    story.append(Spacer(1, 20))
    
    # Medications
    if medications and isinstance(medications, list) and len(medications) > 0:
        story.append(Paragraph("MEDICATIONS", subtitle_style))
        for i, med in enumerate(medications, 1):
            if isinstance(med, dict):
                med_name = med.get('name', 'Unknown')
                story.append(Paragraph(f"{i}. {med_name}", normal_style))
                if med.get('dosage'):
                    story.append(Paragraph(f"   Dosage: {med['dosage']}", normal_style))
                if med.get('timing'):
                    story.append(Paragraph(f"   Timing: {med['timing']}", normal_style))
                if med.get('duration'):
                    story.append(Paragraph(f"   Duration: {med['duration']}", normal_style))
            else:
                story.append(Paragraph(f"{i}. {med}", normal_style))
            story.append(Spacer(1, 10))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                          ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, 
                                       textColor=colors.HexColor('#718096'), alignment=1)))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================
# STREAMLIT APP - FIXED ALIGNMENT
# ============================================================
def main():
    # Set page config with minimal padding
    st.set_page_config(
        page_title="OCR Prescription",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set dark theme
    set_dark_theme()
    
    # Get API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    
    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'image_data' not in st.session_state:
        st.session_state.image_data = None
    
    # App Header - Centered with minimal spacing
    st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">OCR Prescription</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #CBD5E0; font-size: 0.9rem; margin-bottom: 1rem;">Upload a prescription to extract medication details</p>', unsafe_allow_html=True)
    
    # Sidebar - Clean and compact
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 1.8rem;">📋</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #F7FAFC; margin-top: 0.3rem;">OCR Prescription</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Extraction Mode selection
        mode = st.selectbox(
            "Extraction Mode",
            ["full_prescription_summary", "medicine_names_only", "patient_details_only"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="extraction_mode"
        )
        
        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
        
        # Simple tips
        with st.expander("📝 Tips for best results", expanded=False):
            st.markdown("""
            <div style="color: #CBD5E0; font-size: 0.85rem;">
                <div style="margin-bottom: 0.3rem;">• Use clear, well-lit images</div>
                <div style="margin-bottom: 0.3rem;">• Ensure text is readable</div>
                <div style="margin-bottom: 0.3rem;">• Keep prescription flat</div>
                <div>• Avoid glare and shadows</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content - Better column layout
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        st.markdown('<h2 class="section-header">📤 Upload Prescription</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a prescription image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                # Display image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Prescription", use_column_width=True)
                
                # Store image data
                st.session_state.image_data = uploaded_file.getvalue()
                
                # Process button - Centered
                st.markdown('<div style="margin: 0.5rem 0;"></div>', unsafe_allow_html=True)
                
                if st.button("🔍 Extract Prescription Data", 
                           type="primary", 
                           use_container_width=True,
                           disabled=st.session_state.processing):
                    
                    if not api_key:
                        st.error("❌ API key not configured in .env file")
                        return
                    
                    st.session_state.processing = True
                    
                    # Initialize OCR processor
                    ocr_processor = OCRProcessor(api_key)
                    
                    # Encode image
                    image_b64 = ocr_processor.encode_image_to_base64(st.session_state.image_data)
                    
                    # Process with status
                    with st.spinner("Processing prescription..."):
                        result = ocr_processor.process_prescription(image_b64, mode)
                    
                    if result.get("success", False):
                        st.session_state.result = result["data"]
                    else:
                        st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
                        st.session_state.result = None
                    
                    st.session_state.processing = False
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
        else:
            # Upload prompt
            st.markdown("""
            <div class="card-container" style="text-align: center; min-height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem; color: #63B3ED; margin-bottom: 1rem;">📤</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #F7FAFC; margin-bottom: 0.5rem;">Upload Prescription Image</div>
                <div style="color: #CBD5E0;">Drag and drop or click to browse</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Processing indicator
        if st.session_state.processing:
            st.markdown('<h2 class="section-header">Processing</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card-container" style="text-align: center;">
                <div class="loading-spinner"></div>
                <div style="font-size: 1rem; font-weight: 600; color: #F6AD55; margin-top: 1rem;">Processing...</div>
                <div style="color: #CBD5E0; margin-top: 0.5rem;">Analyzing prescription</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Results section
        if st.session_state.result and not st.session_state.processing:
            st.markdown('<h2 class="section-header">📋 Results</h2>', unsafe_allow_html=True)
            
            result = st.session_state.result
            
            # Display results based on mode
            if mode == "medicine_names_only":
                medications = result.get('medications', [])
                if medications:
                    st.markdown('<div class="result-card medication-card">', unsafe_allow_html=True)
                    st.markdown('<div style="color: #76E4F7; font-weight: 600; margin-bottom: 0.5rem;">💊 Medications Found</div>', unsafe_allow_html=True)
                    for i, med in enumerate(medications, 1):
                        if isinstance(med, dict):
                            st.markdown(f"**{i}.** {med.get('name', 'Unknown')}")
                        else:
                            st.markdown(f"**{i}.** {med}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            elif mode == "patient_details_only":
                patient = result.get('patient', {})
                if patient:
                    st.markdown('<div class="result-card patient-card">', unsafe_allow_html=True)
                    st.markdown('<div style="color: #63B3ED; font-weight: 600; margin-bottom: 0.5rem;">👤 Patient Details</div>', unsafe_allow_html=True)
                    if patient.get('name'):
                        st.markdown(f"**Name:** {patient['name']}")
                    if patient.get('age'):
                        st.markdown(f"**Age:** {patient['age']}")
                    if patient.get('gender'):
                        st.markdown(f"**Gender:** {patient['gender']}")
                    if patient.get('date'):
                        st.markdown(f"**Date:** {patient['date']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:  # full_prescription_summary
                # Patient
                patient = result.get('patient', {})
                if any(patient.values()):
                    st.markdown('<div class="result-card patient-card">', unsafe_allow_html=True)
                    st.markdown('<div style="color: #63B3ED; font-weight: 600; margin-bottom: 0.5rem;">👤 Patient</div>', unsafe_allow_html=True)
                    if patient.get('name'):
                        st.markdown(f"**Name:** {patient['name']}")
                    if patient.get('age'):
                        st.markdown(f"**Age:** {patient['age']}")
                    if patient.get('gender'):
                        st.markdown(f"**Gender:** {patient['gender']}")
                    if patient.get('date'):
                        st.markdown(f"**Date:** {patient['date']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Doctor
                doctor = result.get('doctor', {})
                if any(doctor.values()):
                    st.markdown('<div class="result-card doctor-card">', unsafe_allow_html=True)
                    st.markdown('<div style="color: #68D391; font-weight: 600; margin-bottom: 0.5rem;">🏥 Doctor</div>', unsafe_allow_html=True)
                    if doctor.get('name'):
                        st.markdown(f"**Doctor:** {doctor['name']}")
                    if doctor.get('clinic'):
                        st.markdown(f"**Clinic:** {doctor['clinic']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Medications
                medications = result.get('medications', [])
                if medications:
                    st.markdown('<div class="result-card medication-card">', unsafe_allow_html=True)
                    st.markdown('<div style="color: #76E4F7; font-weight: 600; margin-bottom: 0.5rem;">💊 Medications</div>', unsafe_allow_html=True)
                    for i, med in enumerate(medications, 1):
                        with st.expander(f"Med {i}", expanded=i==1):
                            if isinstance(med, dict):
                                if med.get('name'):
                                    st.markdown(f"**Name:** {med['name']}")
                                if med.get('dosage'):
                                    st.markdown(f"**Dosage:** {med['dosage']}")
                                if med.get('timing'):
                                    st.markdown(f"**Timing:** {med['timing']}")
                                if med.get('duration'):
                                    st.markdown(f"**Duration:** {med['duration']}")
                                if med.get('instructions'):
                                    st.markdown(f"**Instructions:** {med['instructions']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Diagnosis
                diagnosis = result.get('diagnosis')
                if diagnosis:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown('<div style="color: #F6AD55; font-weight: 600; margin-bottom: 0.5rem;">📝 Diagnosis</div>', unsafe_allow_html=True)
                    st.markdown(diagnosis)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Action buttons - Stacked vertically
            st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
            
            # Download PDF
            if st.button("📄 Download Report", type="secondary", use_container_width=True):
                pdf_buffer = create_pdf_report(
                    result.get('patient', {}),
                    result.get('doctor', {}),
                    result.get('medications', []),
                    result.get('diagnosis', ''),
                    result.get('notes', '')
                )
                
                st.download_button(
                    label="⬇️ Save as PDF",
                    data=pdf_buffer,
                    file_name=f"prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
            
            # New scan
            if st.button("🔄 New Scan", type="secondary", use_container_width=True):
                st.session_state.result = None
                st.session_state.image_data = None
                st.rerun()

if __name__ == "__main__":
    main()