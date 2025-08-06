import streamlit as st
import pandas as pd
from PIL import Image
import zipfile
import os
import tempfile
import io
from datetime import datetime
import base64

# Import your classifier
from sari_classifier import SariProductionClassifier

# Page config
st.set_page_config(
    page_title="🕺 Sari Quality Checker",
    page_icon="🕺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF6B6B;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 1rem;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.accepted-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 5px solid #00D4AA;
}
.rejected-card {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 5px solid #FF6B6B;
}
.instructions {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 1rem;
    border: 2px dashed #dee2e6;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    """Load the classifier (cached for performance)"""
    return SariProductionClassifier()

def create_thumbnail(image, size=(120, 120)):
    """Create thumbnail for display"""
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    return thumbnail

def process_zip_file(zip_file):
    """Extract images from uploaded zip file"""
    images = []
    filenames = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded zip
        zip_path = os.path.join(temp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getvalue())
        
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find and load images
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    try:
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path).convert('RGB')
                        images.append(img)
                        filenames.append(file)
                    except Exception as e:
                        st.warning(f"⚠️ Couldn't load {file}: {e}")
    
    return images, filenames

def display_results_grid(results, images, filenames):
    """Display results in a nice grid format"""
    
    # Separate accepted and rejected
    accepted_results = results[results['prediction'] == 'Accept']
    rejected_results = results[results['prediction'] == 'Reject']
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Accepted Saris")
        if len(accepted_results) > 0:
            # Create grid layout
            cols = st.columns(3)
            for idx, (_, row) in enumerate(accepted_results.iterrows()):
                col_idx = idx % 3
                with cols[col_idx]:
                    # Find corresponding image
                    img_idx = filenames.index(row['filename'])
                    thumbnail = create_thumbnail(images[img_idx])
                    
                    st.image(thumbnail, use_column_width=True)
                    st.write(f"**{row['filename']}**")
                    st.write(f"Score: {row['final_score']:.3f}")
                    
                    # Color-coded confidence
                    conf = row['confidence']
                    if conf > 0.7:
                        st.success(f"Confidence: High ({conf:.2f})")
                    elif conf > 0.4:
                        st.info(f"Confidence: Medium ({conf:.2f})")
                    else:
                        st.warning(f"Confidence: Low ({conf:.2f})")
        else:
            st.info("No saris met the acceptance criteria.")
    
    with col2:
        st.markdown("### ❌ Rejected Saris")
        if len(rejected_results) > 0:
            # Create grid layout
            cols = st.columns(3)
            for idx, (_, row) in enumerate(rejected_results.iterrows()):
                col_idx = idx % 3
                with cols[col_idx]:
                    # Find corresponding image
                    img_idx = filenames.index(row['filename'])
                    thumbnail = create_thumbnail(images[img_idx])
                    
                    st.image(thumbnail, use_column_width=True)
                    st.write(f"**{row['filename']}**")
                    st.write(f"Score: {row['final_score']:.3f}")
                    
                    # Color-coded confidence
                    conf = row['confidence']
                    if conf > 0.7:
                        st.error(f"Confidence: High ({conf:.2f})")
                    elif conf > 0.4:
                        st.warning(f"Confidence: Medium ({conf:.2f})")
                    else:
                        st.info(f"Confidence: Low ({conf:.2f})")
        else:
            st.info("No saris were rejected.")

def main():
    # Header
    st.markdown('<h1 class="main-header">🕺 Sari Quality Checker</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; margin-bottom: 2rem; color: #666;'>
    Upload sari images to get instant AI-powered quality assessment with confidence scores!
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        with st.spinner("🤖 Loading AI model... This may take a minute..."):
            try:
                st.session_state.classifier = load_classifier()
                st.success("✅ AI model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.stop()
    
    classifier = st.session_state.classifier
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Quality Settings")
        
        # Model settings with better descriptions
        st.markdown("**Acceptance Threshold**")
        threshold = st.slider(
            "Quality Standard", 
            0.0, 1.0, 0.4, 0.05,
            help="Higher = More strict quality filtering"
        )
        
        st.markdown("**Analysis Focus**")
        blend_weight = st.slider(
            "Visual vs Text Matching", 
            0.0, 1.0, 0.8, 0.1,
            help="1.0 = Pure visual similarity, 0.0 = Pure text description matching"
        )
        
        st.markdown("**Quality Description**")
        custom_prompt = st.text_area(
            "Describe Ideal Sari", 
            value="A red sari with evenly distributed floral and geometric motifs",
            help="Describe the characteristics of saris you want to accept"
        )
        
        if st.button("🔧 Update Settings", type="primary"):
            classifier.update_settings(blend_weight, threshold, custom_prompt)
            st.success("✅ Settings updated!")
        
        # Current settings display
        st.markdown("---")
        st.markdown("**Current Settings:**")
        st.markdown(f"- Threshold: `{threshold}`")
        st.markdown(f"- Visual Focus: `{blend_weight}`")
        
        # Instructions
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("- Use well-lit, clear photos")
        st.markdown("- Higher threshold = stricter")
        st.markdown("- Adjust prompt for different styles")
    
    # Main upload section
    st.header("📁 Upload Your Sari Images")
    
    upload_option = st.radio(
        "Choose upload method:", 
        ["📸 Individual Images", "🗂️ Multiple Files", "📁 Zip File"],
        horizontal=True
    )
    
    uploaded_files = []
    
    if upload_option == "📸 Individual Images":
        uploaded_file = st.file_uploader(
            "Choose a sari image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a single high-quality sari image"
        )
        if uploaded_file:
            uploaded_files = [uploaded_file]
    
    elif upload_option == "🗂️ Multiple Files":
        uploaded_files = st.file_uploader(
            "Choose multiple sari images", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple sari images at once (up to 50 recommended)"
        )
    
    elif upload_option == "📁 Zip File":
        zip_file = st.file_uploader(
            "Upload zip file containing sari images", 
            type=['zip'],
            help="Upload a ZIP file with sari images inside"
        )
        if zip_file:
            with st.spinner("📦 Extracting images from ZIP..."):
                try:
                    images, filenames = process_zip_file(zip_file)
                    uploaded_files = list(zip(images, filenames))
                    st.success(f"✅ Extracted {len(images)} images from ZIP")
                except Exception as e:
                    st.error(f"❌ Error extracting ZIP: {e}")
                    uploaded_files = []
    
    # Process uploaded files
    if uploaded_files:
        st.header("🔄 AI Analysis Results")
        
        # Update settings
        classifier.update_settings(blend_weight, threshold, custom_prompt)
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Prepare images and names
            images = []
            names = []
            
            status_text.text("📥 Loading images...")
            progress_bar.progress(20)
            
            if upload_option == "📁 Zip File":
                # Images already extracted
                images = [item[0] for item in uploaded_files]
                names = [item[1] for item in uploaded_files]
            else:
                # Process uploaded files
                for file in uploaded_files:
                    image = Image.open(file).convert('RGB')
                    images.append(image)
                    names.append(file.name)
            
            status_text.text("🧠 Running AI classification...")
            progress_bar.progress(60)
            
            # Classify images
            results = classifier.classify_images(images, names)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Get summary stats
            stats = classifier.get_summary_stats(results)
            
            # Display summary metrics with beautiful cards
            st.markdown("### 📊 Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📸 Total</h3>
                    <h2>{stats['total_images']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>✅ Accepted</h3>
                    <h2>{stats['accepted']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>❌ Rejected</h3>
                    <h2>{stats['rejected']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Avg Score</h3>
                    <h2>{stats['average_score']:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Display acceptance rate
            acceptance_rate = stats['acceptance_rate'] * 100
            st.markdown(f"""
            <div style='text-align: center; font-size: 1.5rem; margin: 1rem 0;'>
            <strong>Acceptance Rate: {acceptance_rate:.1f}%</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Display results grid
            display_results_grid(results, images, names)
            
            # Download section
            st.header("📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV download
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📊 Download Full Report (CSV)",
                    data=csv,
                    file_name=f"sari_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col2:
                # Accepted images summary
                accepted_csv = results[results['prediction'] == 'Accept'].to_csv(index=False)
                st.download_button(
                    label="✅ Download Accepted List",
                    data=accepted_csv,
                    file_name=f"accepted_saris_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Rejected images summary
                rejected_csv = results[results['prediction'] == 'Reject'].to_csv(index=False)
                st.download_button(
                    label="❌ Download Rejected List",
                    data=rejected_csv,
                    file_name=f"rejected_saris_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Detailed results table
            with st.expander("📋 View Detailed Results Table"):
                st.dataframe(
                    results[['filename', 'final_score', 'status', 'confidence']],
                    use_container_width=True,
                    hide_index=True
                )
            
            # Performance insights
            with st.expander("📈 Analysis Insights"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Score Distribution:**")
                    score_ranges = {
                        'Excellent (0.8-1.0)': len(results[results['final_score'] >= 0.8]),
                        'Good (0.6-0.8)': len(results[(results['final_score'] >= 0.6) & (results['final_score'] < 0.8)]),
                        'Fair (0.4-0.6)': len(results[(results['final_score'] >= 0.4) & (results['final_score'] < 0.6)]),
                        'Poor (0.0-0.4)': len(results[results['final_score'] < 0.4])
                    }
                    
                    for range_name, count in score_ranges.items():
                        st.write(f"- {range_name}: {count} images")
                
                with col2:
                    st.markdown("**Recommendations:**")
                    if stats['acceptance_rate'] > 0.8:
                        st.info("🎉 Excellent batch! Most saris meet quality standards.")
                    elif stats['acceptance_rate'] > 0.6:
                        st.info("👍 Good batch! Consider reviewing rejected items.")
                    elif stats['acceptance_rate'] > 0.4:
                        st.warning("⚠️ Mixed quality. Consider tighter photo standards.")
                    else:
                        st.warning("📸 Low acceptance rate. Check photo quality and lighting.")
        
        except Exception as e:
            st.error(f"❌ Error processing images: {str(e)}")
            st.error("Please check your images and try again.")
            st.markdown("**Common issues:**")
            st.markdown("- Corrupted image files")
            st.markdown("- Unsupported file formats")
            st.markdown("- Very large file sizes")
    
    else:
        # Instructions when no files uploaded
        st.markdown("""
        <div class="instructions">
        <h3>📖 How to Use the Sari Quality Checker</h3>
        
        <h4>Step 1: Upload Images</h4>
        <ul>
        <li><strong>Individual Images:</strong> Upload one sari at a time for quick checks</li>
        <li><strong>Multiple Files:</strong> Select multiple sari images at once (recommended for small batches)</li>
        <li><strong>Zip File:</strong> Upload a ZIP containing many sari images (best for large batches)</li>
        </ul>
        
        <h4>Step 2: Adjust Quality Settings (Optional)</h4>
        <ul>
        <li><strong>Quality Standard:</strong> Higher threshold = more strict filtering</li>
        <li><strong>Analysis Focus:</strong> Balance between visual similarity and text description matching</li>
        <li><strong>Quality Description:</strong> Customize what makes an ideal sari</li>
        </ul>
        
        <h4>Step 3: Review Results</h4>
        <ul>
        <li>See accepted and rejected saris with confidence scores</li>
        <li>Download CSV reports for your records</li>
        <li>Use insights to improve your sari photography</li>
        </ul>
        
        <h4>💡 Pro Tips for Best Results:</h4>
        <ul>
        <li>Use high-resolution, well-lit photographs</li>
        <li>Ensure saris are fully visible and flat</li>
        <li>Avoid blurry or dark images</li>
        <li>Consistent lighting across all photos works best</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo section
        st.markdown("### 🎮 Try It Now!")
        st.markdown("Upload some sari images above to see the AI classifier in action!")

if __name__ == "__main__":
    main()
