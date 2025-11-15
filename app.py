import streamlit as st
from datetime import datetime
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from collections import Counter

from model.predict import predict, model, transform  # assuming these exist
from model.gradcam import generate_cam
from llm.advisor import (
    generate_advice,
    chat_with_advisor,
    generate_multilingual_summary,
    compare_with_similar_diseases,
    generate_seasonal_tips
)

# Page Configuration
st.set_page_config(
    page_title="AgriSense-X: AI Crop Doctor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #2E7D32; text-align: center; margin-bottom: 2rem; }
    .disease-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; color: white; margin: 10px 0;
    }
    .confidence-high { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; }
    .confidence-medium { background-color: #FF9800; color: white; padding: 10px; border-radius: 5px; }
    .confidence-low { background-color: #F44336; color: white; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return model, transform


model, transform = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_diagnosis' not in st.session_state:
    st.session_state.current_diagnosis = None

# Sidebar
with st.sidebar:
    st.title("🌾 AgriSense-X")
    st.markdown("---")

    language = st.selectbox(
        "🌍 Language / भाषा",
        ["english", "hindi", "nepali"],
        format_func=lambda x: {
            "english": "English", "hindi": "हिंदी (Hindi)", "nepali": "नेपाली (Nepali)"}[x]
    )

    st.markdown("---")

    # Farm Analytics
    if st.session_state.history:
        st.subheader("📊 Your Farm Analytics")

        diseases = [h['disease'] for h in st.session_state.history]
        disease_counts = Counter(diseases)

        fig = px.pie(values=list(disease_counts.values()), names=list(
            disease_counts.keys()), title="Disease Distribution")
        st.plotly_chart(fig, use_container_width=True)

        confidences = [h['confidence'] for h in st.session_state.history]
        dates = [h['timestamp'] for h in st.session_state.history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=confidences,
                      mode='lines+markers', name='Confidence'))
        fig.update_layout(title="Detection Confidence Over Time")
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Total Scans", len(st.session_state.history))
        st.metric("Avg Confidence",
                  f"{sum(confidences)/len(confidences):.1%}" if confidences else "0%")

    st.markdown("---")

    if st.button("📅 This Month's Tips"):
        with st.spinner("Fetching seasonal advice..."):
            tips = generate_seasonal_tips(
                "General crops", datetime.now().month)
            st.info(tips)

    st.markdown("---")

    if st.session_state.history and st.button("📥 Export History"):
        df = pd.DataFrame(st.session_state.history)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"agrisense_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Main Content
st.markdown("<h1 class='main-header'>🌱 AgriSense-X: Your AI Crop Doctor</h1>",
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Diagnosis", "💬 Ask Expert"])

# Tab 1: Diagnosis
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Take a photo or upload image", type=["jpg", "png", "jpeg"],
            help="Take a clear photo of the affected leaf in good lighting"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔬 Analyze Disease", type="primary"):
                with st.spinner("🧬 AI is analyzing the leaf..."):
                    uploaded_file.seek(0)
                    with open("temp.jpg", "wb") as f:
                        f.write(uploaded_file.read())

                    label, conf = predict("temp.jpg")

                    parts = label.split('___')
                    crop = parts[0] if len(parts) > 1 else "Unknown"
                    disease = parts[1] if len(parts) > 1 else label

                    st.session_state.current_diagnosis = {
                        'disease': disease,
                        'crop': crop,
                        'confidence': conf,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'image_path': 'temp.jpg',
                        'gradcam_image': None,
                        'summary': None,
                        'advice': None
                    }

                    st.session_state.history.append(
                        st.session_state.current_diagnosis.copy())
                    st.rerun()

    with col2:
        if st.session_state.current_diagnosis:
            diag = st.session_state.current_diagnosis
            conf = diag['confidence']

            if conf >= 0.8:
                conf_class, emoji, text = "confidence-high", "✅", "High Confidence"
            elif conf >= 0.6:
                conf_class, emoji, text = "confidence-medium", "⚠️", "Medium Confidence"
            else:
                conf_class, emoji, text = "confidence-low", "❗", "Low Confidence"

            st.subheader("🦠 Diagnosis Results")
            st.markdown(f"""
            <div class='disease-card'>
                <h2>🌿 {diag['crop']}</h2>
                <h3>🦠 {diag['disease']}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='{conf_class}'>
                <h3>{emoji} {text}: {conf:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(conf)

            if conf < 0.6:
                st.warning(
                    "⚠️ Low confidence detection. Consider expert inspection.")
                with st.expander("🔍 See Similar Diseases"):
                    with st.spinner("Analyzing similar conditions..."):
                        differential = compare_with_similar_diseases(
                            diag['disease'], conf)
                        st.markdown(differential)

    # ── Detailed analysis sections (hidden by default in expanders)
    if st.session_state.current_diagnosis:
        st.markdown("---")

        # GradCAM Expander
        with st.expander("🔥 AI Focus Areas (GradCAM)", expanded=False):
            if 'gradcam_image' not in diag or diag['gradcam_image'] is None:
                with st.spinner("Generating visualization..."):
                    try:
                        img_tensor = transform(image).unsqueeze(0)
                        target_class = [i for i, c in enumerate(
                            model.classes) if c == diag['disease']][0] if hasattr(model, 'classes') else 0
                        cam_image = generate_cam(
                            model, img_tensor, target_class)
                        diag['gradcam_image'] = cam_image
                    except Exception as e:
                        st.error(f"GradCAM failed: {str(e)}")
                        diag['gradcam_image'] = None

            if diag['gradcam_image'] is not None:
                st.image(
                    diag['gradcam_image'],
                    caption="Red = high attention | Blue = low attention",
                    use_container_width=True
                )
            else:
                st.info("GradCAM visualization not available.")

        # Quick Summary Expander
        with st.expander(f"🌍 Quick Summary ({language})", expanded=False):
            if 'summary' not in diag or diag['summary'] is None:
                with st.spinner("Translating..."):
                    diag['summary'] = generate_multilingual_summary(
                        diag['disease'], language)
            st.success(diag['summary'])

        # Comprehensive Advisory Expander
        with st.expander("📋 Comprehensive Agricultural Advisory", expanded=False):
            if 'advice' not in diag or diag['advice'] is None:
                with st.spinner("🤖 Generating expert advice..."):
                    diag['advice'] = generate_advice(
                        diag['crop'],
                        diag['disease'],
                        diag['confidence'],
                        language
                    )
            st.markdown(diag['advice'])

# Tab 2: Chat with Expert
with tab2:
    st.subheader("💬 Chat with Agricultural Expert")

    if not st.session_state.current_diagnosis:
        st.info(
            "👆 Please upload and analyze an image first to start chatting about your diagnosis.")
    else:
        st.success(
            f"Currently discussing: **{st.session_state.current_diagnosis['disease']}** on **{st.session_state.current_diagnosis['crop']}**")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask anything about the disease, treatment, costs, etc..."):
            # Add and show user message
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and show assistant response
            with st.chat_message("assistant"):
                with st.spinner("Expert is thinking..."):
                    response = chat_with_advisor(
                        prompt,
                        st.session_state.current_diagnosis,
                        st.session_state.chat_history[:-1]
                    )
                st.markdown(response)

            # Save assistant response
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

        # Optional: Quick questions
        st.markdown("---")
        st.markdown("**Quick Questions:**")
        quick_questions = [
            "What's the cheapest treatment option?",
            "How long until I see improvement?",
            "Is this disease contagious to other plants?",
            "Can I still harvest affected plants?"
        ]

        cols = st.columns(2)
        for i, q in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(q, key=f"quick_{i}"):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": q})
                    with st.chat_message("user"):
                        st.markdown(q)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = chat_with_advisor(
                                q,
                                st.session_state.current_diagnosis,
                                st.session_state.chat_history[:-1]
                            )
                        st.markdown(response)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>For educational and advisory purposes only. Always consult local agricultural experts for critical decisions.</p>
</div>
""", unsafe_allow_html=True)
