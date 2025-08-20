# Glacier Melting — Portfolio Streamlit App
# Single-file Streamlit app combining a colorful personal portfolio
# and a computer-vision pipeline to track glacier melting between
# two images (before/after).
# Run: `streamlit run glacier_portfolio_app.py`

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
import base64
import os
import tempfile
from datetime import datetime

st.set_page_config(page_title="Glacier Portfolio & CV", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- Styles ----------
st.markdown(
    """
    <style>
    .header {background: linear-gradient(90deg,#0f172a,#0ea5e9); padding:30px; border-radius:12px; color:white}
    .sub {color: #e2e8f0}
    .card {background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); padding:18px; border-radius:12px; box-shadow: 0 8px 20px rgba(2,6,23,0.4)}
    .sm {font-size:0.9rem; color:#cbd5e1}
      .sub{color:green;font-size:large}

    </style>
    """, unsafe_allow_html=True)

# ---------- Header / Portfolio ----------
with st.container():
    left, right = st.columns([3, 1])
    with left:
        st.header("🧊 Glacier Melting Tracker Using Computer Vision")
        st.markdown('<div class="header">', unsafe_allow_html=True)
        st.title("Vishesh Kumar Prajapati — Computer Vision & Data Science")
        st.markdown("<p  class='sub' >Portfolio · Computer Vision · Python · Streamlit</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.header("About Me")
        st.write(
            "I build interactive visualisations, CV pipelines and deploy them as web apps. This single-file Streamlit app demonstrates a colorful portfolio section and a glacier melting tracker using OpenCV.")
        st.markdown("**Skills:** Python, OpenCV, PyTorch (optional), Streamlit, Remote Sensing, Data Viz")
        st.markdown("---")
        st.header("Projects & Contacts")
        st.markdown(
            "- **Glacier Melting Tracker** — Image alignment, differencing, optional segmentation, interactive before/after slider.")
        st.markdown("- **Other projects** — Weather bot, Deforestation sensor, Movie UI, Salary prediction model.")
    st.subheader("📬 Contact Details")

    st.markdown("""
      [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/vishesh-kumar-prajapati-45111829a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)  
      [![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/vishes-i/future-interns/commit/ddc7ef65bcf8417b718241c2fe8d6dd715d8a8b6)  
      [![Email](https://img.shields.io/badge/Email-Send-red?logo=gmail)](mailto:Visheshprajapati7920@gmail.com)
      """)
    with right:
        st.image(
            r"C:\Users\adity\OneDrive\Documents\Pictures\Documents - Copy\1685670270272.jpg",  # Placeholder image URL
            caption="SHIVSHAKTI",
            width=200
        )

st.markdown("---")

# ---------- Sidebar controls ----------
st.sidebar.title("Tracker Controls")
alignment_algo = st.sidebar.selectbox("Alignment algorithm", options=["ORB+Homography", "ECC (if available)"], index=0)
use_segmentation = st.sidebar.checkbox("Try semantic segmentation (DeepLabV3 if available)", value=False)
colormap = st.sidebar.selectbox("Diff colormap", options=["JET", "INFERNO", "HOT", "PLASMA"], index=0)
thresholding = st.sidebar.slider("Diff threshold (for mask)", 1, 255, 30)


# ---------- Utility functions ----------

def pil_to_cv(img_pil):
    arr = np.array(img_pil.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv):
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def read_image(uploaded_file):
    image = Image.open(uploaded_file)
    return image


def resize_keeping_aspect(img_cv, max_dim=1200):
    h, w = img_cv.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_cv


# ---------- Alignment functions ----------

def align_orb(img1, img2, max_features=5000, good_match_percent=0.15):
    """Align img2 to img1 using ORB + Homography."""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(max_features)
    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k2, d2 = orb.detectAndCompute(img2_gray, None)
    if d1 is None or d2 is None:
        return img2, None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)
    numGood = int(len(matches) * good_match_percent)
    matches = matches[:max(numGood, 4)]
    if len(matches) < 4:
        return img2, None
    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, m in enumerate(matches):
        pts1[i, :] = k1[m.queryIdx].pt
        pts2[i, :] = k2[m.trainIdx].pt
    m, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if m is None:
        return img2, None
    h, w = img1.shape[:2]
    aligned = cv2.warpPerspective(img2, m, (w, h))
    return aligned, m


def compute_diff(img1, img2, thresh=30):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    d = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
    return d, mask


def apply_colormap(diff_gray, cmap_name="JET"):
    cmap_map = {
        'JET': cv2.COLORMAP_JET,
        'INFERNO': cv2.COLORMAP_INFERNO,
        'HOT': cv2.COLORMAP_HOT,
        'PLASMA': cv2.COLORMAP_PLASMA
    }
    cmap = cmap_map.get(cmap_name, cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(diff_gray, cmap)
    return colored


# Optional semantic segmentation loader
seg_model = None
if use_segmentation:
    try:
        import torch
        import torchvision

        seg_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).eval()
        st.sidebar.success("Loaded DeepLabV3 for segmentation (CPU).")
    except Exception as e:
        st.sidebar.warning("Could not load segmentation model: {}".format(e))
        seg_model = None


def segment_image_pil(pil_img):
    """Return a mask (0/255) from DeepLabV3 for 'person'/'background' style segmentation - good for landscapes too."""
    if seg_model is None:
        return None
    img = pil_img.convert('RGB')
    tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    input_tensor = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = seg_model(input_tensor)['out'][0]
    labels = out.argmax(0).byte().cpu().numpy()
    # Convert labels to binary mask: any label > 0 considered foreground
    mask = (labels > 0).astype(np.uint8) * 255
    return mask


# ---------- Interface: Upload images ----------
st.header("Glacier Melting Tracker")
st.write(
    "Upload a *BEFORE* and *AFTER* image (satellite / drone / photo). The app will align, compute differences and show an interactive before/after comparison.")

col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("Upload BEFORE image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key='before')
with col2:
    after_file = st.file_uploader("Upload AFTER image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key='after')

use_sample = False
if not before_file or not after_file:
    st.info("You can try sample glacier images if you don't have your own. (Small images for demo.)")
    if st.button("Load sample images"):
        use_sample = True
        # Load from package - create two synthetic demo images
        demo_dir = tempfile.gettempdir()
        b = Image.new('RGB', (900, 600), (150, 180, 220))
        a = Image.new('RGB', (900, 600), (150, 180, 220))
        # Draw a white 'ice' blob that shrinks in AFTER
        cvb = pil_to_cv(b)
        cva = pil_to_cv(a)
        cv2.circle(cvb, (450, 300), 200, (240, 240, 255), -1)
        cv2.circle(cva, (450, 300), 150, (240, 240, 255), -1)
        before_img = cvb
        after_img = cva
    else:
        before_img = None
        after_img = None
else:
    before_img = pil_to_cv(read_image(before_file))
    after_img = pil_to_cv(read_image(after_file))

if before_img is not None and after_img is not None:
    # Resize to reasonable size
    before_img = resize_keeping_aspect(before_img, max_dim=1200)
    after_img = resize_keeping_aspect(after_img, max_dim=1200)

    st.markdown("### Alignment & Difference")
    colA, colB = st.columns([1, 1])
    with colA:
        st.image(cv_to_pil(before_img), caption='BEFORE (aligned to original orientation)')
    with colB:
        st.image(cv_to_pil(after_img), caption='AFTER (raw upload)')

    # Align
    if alignment_algo == "ORB+Homography":
        aligned_after, H = align_orb(before_img, after_img)
    else:
        aligned_after, H = align_orb(before_img, after_img)

    if H is None:
        st.warning(
            "Could not compute a robust homography — showing unaligned AFTER image. Try images with overlapping regions or different algorithm settings.")
        aligned_after = after_img
    else:
        st.success("Alignment succeeded. Homography matrix computed.")

    st.write("**Homography (first 3x3 block):**")
    st.code(np.array_str(H, precision=3)) if H is not None else None

    st.markdown("---")
    diff_gray, mask = compute_diff(before_img, aligned_after, thresh=thresholding)
    heat = apply_colormap(diff_gray, cmap_name=colormap)

    # Optionally refine mask with segmentation
    if use_segmentation and seg_model is not None:
        try:
            seg_mask_before = segment_image_pil(cv_to_pil(before_img))
            seg_mask_after = segment_image_pil(cv_to_pil(aligned_after))
            if seg_mask_before is not None and seg_mask_after is not None:
                seg_mask = cv2.bitwise_and(seg_mask_before, seg_mask_after)
                mask = cv2.bitwise_and(mask, seg_mask)
                st.sidebar.info("Refined diff mask using DeepLabV3 segmentation.")
        except Exception as e:
            st.sidebar.warning("Segmentation refinement failed: {}".format(e))

    # Overlay heat on top of BEFORE image for visualization
    overlay = cv2.addWeighted(before_img, 0.6, heat, 0.4, 0)
    overlay_pil = cv_to_pil(overlay)
    heat_pil = cv_to_pil(heat)

    # Display interactive before-after slider via simple HTML
    st.markdown("**Interactive comparison (drag):**")


    # Prepare images as base64
    def pil_to_b64(img_pil):
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


    b64_before = pil_to_b64(cv_to_pil(before_img))
    b64_after = pil_to_b64(overlay_pil)

    slider_html = f"""
    <style>
    .comp-wrap {{width:100%; max-width:1100px; margin:auto}}
    .comp-img {{width:100%; display:block}}
    .comp-slider {{-webkit-appearance: none; width:100%;}}
    </style>
    <div class="comp-wrap">
      <div style="position:relative;">
        <img src="data:image/png;base64,{b64_before}" class="comp-img" id="img1">
        <img src="data:image/png;base64,{b64_after}" class="comp-img" id="img2" style="position:absolute; top:0; left:0; clip:rect(0px,600px,9999px,0px);">
      </div>
      <input type="range" min="0" max="100" value="50" id="s" class="comp-slider">
    </div>
    <script>
    const s = document.getElementById('s');
    const img2 = document.getElementById('img2');
    s.oninput = function(){{
      const val = this.value/100.0;
      const w = img2.naturalWidth;
      const clipx = Math.round(w * val);
      img2.style.clip = 'rect(0px,'+clipx+'px,9999px,0px)';
    }}
    </script>
    """

    st.components.v1.html(slider_html, height=520)

    st.markdown("---")
    st.subheader("Diff Mask & Statistics")
    st.image(cv_to_pil(mask), caption='Binary diff mask (areas of change)')
    # Statistics: percentage changed
    pct = (np.count_nonzero(mask) / mask.size) * 100.0
    st.metric("Percent changed (approx)", f"{pct:.4f}%")


    # Download result as ZIP with images & mask
    def make_download_zip():
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            for name, img in [('before.png', cv_to_pil(before_img)), ('after_aligned.png', cv_to_pil(aligned_after)),
                              ('overlay.png', overlay_pil), ('mask.png', cv_to_pil(mask))]:
                b = io.BytesIO()
                img.save(b, format='PNG')
                z.writestr(name, b.getvalue())
        return buf.getvalue()


    zipped = make_download_zip()
    st.download_button("Download results (zip)", data=zipped,
                       file_name=f"glacier_results_{datetime.now().strftime('%Y%m%d_%H%M')}.zip")

else:
    st.info("Upload both BEFORE and AFTER images (or load sample) to run the tracker.")
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

frames = np.arange(1, 21)
np.random.seed(42)
initial_area = 1000
melting = np.cumsum(np.random.uniform(5, 20, size=20))
areas = np.clip(initial_area - melting, 0, None)

fig, ax = plt.subplots()
ax.plot(frames, areas, marker='o', color='blue')
ax.set_title("Simulated Glacier Melting Tracking")
ax.set_xlabel("Frame / Time Step")
ax.set_ylabel("Glacier Area (arbitrary units)")
ax.grid(True)

st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

years = np.arange(2000, 2021)
np.random.seed(123)
initial_value = 1000
annual_change = np.random.uniform(-50, -10, size=len(years))
values = initial_value + np.cumsum(annual_change)
values = np.clip(values, 0, None)

df = pd.DataFrame({"Year": years, "Value": values})

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Year"], df["Value"], marker='o', linestyle='-', color='teal')
ax.set_title("Yearly Tracking Data Flow (Simulated)")
ax.set_xlabel("Year")
ax.set_ylabel("Tracked Value (e.g., Glacier Area)")
ax.grid(True)
plt.xticks(years, rotation=45)
plt.tight_layout()

st.pyplot(fig)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Example images (use your own URLs or local paths)
images = {
    "Glacier Jan": r"C:\Users\adity\Downloads\dNDSI.tif",
    "Glacier Feb": r"C:\Users\adity\Downloads\glacier_melt_diff.png",
    "Glacier Mar": r"C:\Users\adity\Downloads\glacier_melt_detected.png"
}

# Initialize or load view counts in session state
if "views" not in st.session_state:
    st.session_state.views = {key: 0 for key in images.keys()}

st.title("Image Viewer with View Tracking")

# Display images with buttons to simulate "view"
for name, url in images.items():
    st.image(url, caption=name)
    if st.button(f"View {name}"):
        st.session_state.views[name] += 1
        st.success(f"Viewed {name}!")

# Show current view counts
st.subheader("View Counts")
df_views = pd.DataFrame.from_dict(st.session_state.views, orient='index', columns=['Views'])
st.bar_chart(df_views)

# Optionally show raw data table
st.subheader("Raw View Data")
st.dataframe(df_views)
import streamlit as st
import pandas as pd
import plotly.express as px

# Sample glacier location data (latitude, longitude, name)
data = {
    "Name": ["Glacier A", "Glacier B", "Glacier C"],
    "Latitude": [61.5, 46.8, 78.9],
    "Longitude": [-149.9, 11.2, 16.0],
    "Size_km2": [120, 80, 200]
}
df = pd.DataFrame(data)

st.title("World Glacier Map")

fig = px.scatter_geo(df,
                     lat="Latitude",
                     lon="Longitude",
                     hover_name="Name",
                     size="Size_km2",
                     projection="natural earth",
                     title="Sample Glacier Locations")
st.plotly_chart(fig)
# --------------------------------------------------------------

# ---------- Footer / Extras ----------
st.markdown("---")
st.markdown("### Notes & Next steps")
st.write(
    "- This demo performs classical CV alignment and simple differencing. For production-grade glacier change detection, consider using multispectral satellite data (Landsat/Sentinel), radiometric corrections, image co-registration with georeferencing, and specialized change detection algorithms.\n- Want an .ipynb notebook version, or to add a GIS map overlay (georeferenced), or to run GPU-accelerated segmentation? Tell me and I will adapt it.")

# Helpful quick links for running locally
st.markdown(
    "**Run locally:** `pip install streamlit opencv-python pillow numpy` then `streamlit run glacier_portfolio_app.py`")
import streamlit as st

st.title("Meet Our Team")

teammates = [
    {
        "name": "Vishesh Kumar Prajapati",
        "role": "Full Stack Web  Developer ",
        "photo": r"C:\Users\adity\OneDrive\Documents\ddhscghu.jpg",
        "bio": "Expert in computer vision and machine learning."
    },
    {
        "name": "Sumit Yadav",
        "role": "B.tech Cse",
        "photo": r"C:\Users\adity\WebstormProjects\untitled1\sumit.png",
        "bio": "Specializes in building interactive web ."
    },
    {
        "name": "Vijay Kharwar",
        "role": "B.tech Cse",
        "photo": r"C:\Users\adity\OneDrive\Documents\dhj0aerf.jpg",
        "bio": "Keeps the project on track and well-coordinated."
    },
]

for member in teammates:
    st.image(member["photo"], width=100)
    st.markdown(f"### {member['name']}  \n**{member['role']}**  \n{member['bio']}")
    st.write("---")

# End of file
