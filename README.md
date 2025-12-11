# Import necessary libraries
import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import Affine
import geopandas as gpd
import tempfile
import zipfile
import io
import os
from shapely.geometry import shape, mapping
import json

st.set_page_config(layout="wide", page_title="Aplikasi Peta Rawan Longsor (AHP)")

st.title("Aplikasi Peta Rawan Longsor — AHP + Download GeoTIFF / PNG / GeoJSON/SHAPE")

st.markdown("""
Upload:
- *DEM* (.tif) — raster elevasi
- *Curah hujan* (.tif atau .csv)
- *Geologi* (GeoJSON / Shapefile zipped)
Aplikasi akan menghitung kemiringan (slope) dari DEM, menormalisasi tiap kriteria, menghitung bobot AHP, membuat peta risiko, dan menyediakan download.
""")

# -------------------------
# functions 
# -------------------------
def compute_slope(dem_arr, dem_affine):
    # simple slope from gradients (rise/run)
    # convert pixel spacing from affine
    xres = dem_affine.a
    yres = -dem_affine.e
    gy, gx = np.gradient(dem_arr, yres, xres)
    slope = np.sqrt(gx*2 + gy*2)
    return slope

def normalize(arr):
    arr = np.array(arr, dtype=float)
    mask = np.isnan(arr)
    mmin = np.nanmin(arr)
    mmax = np.nanmax(arr)
    if mmax == mmin:
        res = np.zeros_like(arr)
    else:
        res = (arr - mmin) / (mmax - mmin)
    res[mask] = np.nan
    return res

def ahp_weights_from_matrix(mat):
    # mat: square numpy array
    vals, vecs = np.linalg.eig(mat)
    max_idx = np.argmax(vals.real)
    v = vecs[:, max_idx].real
    w = v / v.sum()
    return np.abs(w)

def consistency_ratio(mat):
    # basic consistency ratio for AHP (3x3)
    n = mat.shape[0]
    vals, vecs = np.linalg.eig(mat)
    lambda_max = np.max(vals.real)
    CI = (lambda_max - n) / (n - 1)
    # Random Index (RI) for n
    RI_dict = {1:0.0,2:0.0,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45}
    RI = RI_dict.get(n, 1.45)
    if RI == 0:
        return 0.0
    return CI / RI

def rasterize_vector_to_match(src_shapes, out_shape, transform, attribute_map=None, dtype='float32'):
    # src_shapes: list of (geom, value)
    raster = features.rasterize(
        ((geom, val) for geom,val in src_shapes),
        out_shape=out_shape,
        transform=transform,
        fill=np.nan,
        dtype=dtype
    )
    return raster

def unzip_shapefile_to_temp(uploaded_zip):
    tf = tempfile.TemporaryDirectory()
    z = zipfile.ZipFile(uploaded_zip)
    z.extractall(tf.name)
    # find shapefile
    for fname in os.listdir(tf.name):
        if fname.endswith(".shp"):
            return tf.name, os.path.join(tf.name, fname)
    return tf.name, None

# -------------------------
# Upload inputs
# -------------------------
col1, col2, col3 = st.columns(3)
with col1:
    dem_file = st.file_uploader("Upload DEM (.tif)", type=["tif", "tiff"])

with col2:
    rain_file = st.file_uploader("Upload Curah Hujan (.tif atau .csv)", type=["tif","tiff","csv"])

with col3:
    geo_file = st.file_uploader("Upload Geologi (GeoJSON) atau (ZIPPED Shapefile)", type=["geojson","json","zip","shp"])

# Optional: allow user choose rainfall interpretation if csv
if rain_file and rain_file.name.lower().endswith(".csv"):
    st.info("CSV curah hujan akan diinterpretasikan sebagai tabel titik/raster sederhana; pastikan kolom value atau raster-like.")

# Proceed when DEM provided
if dem_file:
    # Save DEM to a temp file for rasterio
    dem_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    dem_tmp.write(dem_file.getvalue())
    dem_tmp.flush()
    dem_tmp.close()

    dem_src = rasterio.open(dem_tmp.name)
    dem = dem_src.read(1, masked=False).astype(float)
    dem_meta = dem_src.meta.copy()
    dem_transform = dem_src.transform
    dem_crs = dem_src.crs

    st.success("DEM loaded. Resolusi: {} x {}, CRS: {}".format(dem.shape[1], dem.shape[0], dem_crs))

    # Compute slope
    slope = compute_slope(dem, dem_transform)
    slope_norm = normalize(slope)

    # Read rainfall
    if rain_file:
        if rain_file.name.lower().endswith(".csv"):
            # assume CSV contains grid-like data? To keep simple: use single mean value or ask user
            rain_df = pd.read_csv(rain_file)
            # try to find numeric column
            numeric_cols = rain_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                st.error("CSV curah hujan tidak mengandung kolom numerik. Sediakan .tif atau .csv dengan kolom numerik.")
                st.stop()
            # simplest: use mean as uniform raster
            rain_mean = rain_df[numeric_cols[0]].mean()
            rainfall = np.full(dem.shape, rain_mean, dtype=float)
            st.info(f"CSV curah hujan dibaca -> menggunakan mean {rain_mean:.2f} sebagai raster seragam.")
        else:
            # raster
            rain_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            rain_tmp.write(rain_file.getvalue())
            rain_tmp.flush()
            rain_tmp.close()
            with rasterio.open(rain_tmp.name) as rsrc:
                # reproject / resample to DEM shape if needed
                if rsrc.crs != dem_crs or rsrc.width != dem.shape[1] or rsrc.height != dem.shape[0]:
                    # resample to DEM's shape
                    data = rsrc.read(
                        out_shape=(rsrc.count, dem.shape[0], dem.shape[1]),
                        resampling=Resampling.bilinear
                    )[0].astype(float)
                    rainfall = data
                else:
                    rainfall = rsrc.read(1).astype(float)
            st.success("Raster curah hujan dimuat dan disesuaikan ke DEM.")
        rain_norm = normalize(rainfall)
    else:
        rain_norm = np.zeros_like(slope_norm)
        st.warning("Curah hujan belum diunggah. Menggunakan nol untuk curah hujan (boleh upload nanti).")

    # Read geology vector and rasterize
    geology_raster = None
    geology_classes = {}
    if geo_file:
        # handle geojson / shp zip / shp
        if geo_file.name.lower().endswith(".zip"):
            tfdir, shp_path = unzip_shapefile_to_temp(io.BytesIO(geo_file.getvalue()))
            if not shp_path:
                st.error("ZIP tidak berisi shapefile (.shp).")
            gdf = gpd.read_file(shp_path)
        elif geo_file.name.lower().endswith(".shp"):
            # direct shp
            shp_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".shp")
            shp_tmp.write(geo_file.getvalue())
            shp_tmp.flush(); shp_tmp.close()
            gdf = gpd.read_file(shp_tmp.name)
        else:
            # geojson
            geojson_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
            geojson_tmp.write(geo_file.getvalue())
            geojson_tmp.flush(); geojson_tmp.close()
            gdf = gpd.read_file(geojson_tmp.name)

        # ensure gdf CRS matches DEM
        if gdf.crs != dem_crs:
            try:
                gdf = gdf.to_crs(dem_crs)
            except Exception as e:
                st.error("Gagal reproject geologi ke CRS DEM: " + str(e))
                st.stop()

        st.success(f"Geologi berhasil dimuat — {len(gdf)} fitur.")

        # choose attribute to classify geology (if exists)
        candidate_attrs = [c for c in gdf.columns if c not in ('geometry',)]
        attr = None
        if len(candidate_attrs) == 0:
            # assign a single class
            gdf['__class'] = 1
            attr = '__class'
        else:
            attr = candidate_attrs[0]  # pick first attribute (user can change code)
        # Map unique classes to integer values
        unique_vals = list(gdf[attr].unique())
        class_map = {val: i+1 for i,val in enumerate(unique_vals)}
        # Prepare shapes for rasterize
        shapes = []
        for _, row in gdf.iterrows():
            shapes.append((row.geometry, class_map[row[attr]]))
        geology_raster = features.rasterize(
            ((geom, val) for geom,val in shapes),
            out_shape=dem.shape,
            transform=dem_transform,
            fill=0,
            dtype='int16'
        )
        # Show geology classes and let user assign risk scores per class
        st.subheader("Pengaturan Skor Risiko untuk Setiap Kelas Geologi")
        geo_scores = {}
        cols = st.columns(3)
        for i, val in enumerate(unique_vals):
            default = 1.0
            geo_scores[val] = cols[i % 3].slider(f"Skor risiko untuk '{val}'", 0.0, 5.0, default=default, step=0.1)
        # build geology risk raster by mapping class to score
        georisk = np.zeros_like(geology_raster, dtype=float)
        inv_map = {v:k for k,v in class_map.items()}
        for code in np.unique(geology_raster):
            if code == 0:
                georisk[geology_raster == code] = 0.0
            else:
                classname = inv_map.get(code)
                georisk[geology_raster == code] = geo_scores.get(classname, 1.0)
        georisk_norm = normalize(georisk)
    else:
        georisk_norm = np.zeros_like(slope_norm)
        st.warning("Geologi belum diunggah. Menggunakan nol untuk komponen geologi.")

    # -------------------------
    # AHP interface
    # -------------------------
    st.subheader("AHP — Matriks Perbandingan Berpasangan (Slope, Curah Hujan, Geologi)")
    st.write("Masukkan perbandingan (mis: jika Slope 3x lebih penting dibanding Rainfall, isi 3 pada cell Slope vs Rain; untuk sebaliknya masukkan 1/3).")

    # default pairwise matrix for 3 criteria
    # order: [Slope, Rainfall, Geologi]
    default_matrix = np.array([[1,  2,  3],
                               [1/2,1,  2],
                               [1/3,1/2,1]])
    # create inputs
    mat = np.zeros((3,3), dtype=float)
    names = ["Slope", "Rainfall", "Geology"]
    matrix_inputs = {}
    for i in range(3):
        for j in range(3):
            if i == j:
                mat[i,j] = 1.0
            elif i < j:
                key = f"m_{i}_{j}"
                matrix_inputs[key] = st.number_input(f"{names[i]} vs {names[j]}", value=float(default_matrix[i,j]), format="%.3f", key=key)
                mat[i,j] = float(matrix_inputs[key])
            else:
                # lower triangle is reciprocal
                mat[i,j] = 1.0 / mat[j,i]

    st.write("Matriks AHP:")
    st.write(pd.DataFrame(mat, index=names, columns=names))

    # compute weights
    weights = ahp_weights_from_matrix(mat)
    cr = consistency_ratio(mat)
    st.write("Bobot (normalisasi):")
    wdf = pd.DataFrame({"Criteria": names, "Weight": weights})
    st.table(wdf)
    st.write(f"Consistency Ratio (CR): {cr:.3f} (ideal < 0.10)")

    # -------------------------
    # Combine layers
    # -------------------------
    st.subheader("Penggabungan Layer & Visual")
    # Ensure we have three normalized layers: slope_norm, rain_norm, georisk_norm
    # If some are missing, they are zeros (handled earlier)
    # Compute risk map:
    combined = (weights[0] * slope_norm) + (weights[1] * (rain_norm if 'rain_norm' in locals() else np.zeros_like(slope_norm))) + (weights[2] * (georisk_norm if 'georisk_norm' in locals() else np.zeros_like(slope_norm)))
    combined_norm = normalize(combined)

    # Let user choose color map and show layers
    show_dem = st.checkbox("Tampilkan DEM (contour style)", value=False)
    show_rain = st.checkbox("Tampilkan Curah Hujan", value=True)
    show_geo = st.checkbox("Tampilkan Geologi rasterized", value=True)
    show_risk = st.checkbox("Tampilkan Peta Risiko (heatmap)", value=True)

    fig, ax = plt.subplots(figsize=(9,7))
    if show_dem:
        ax.imshow(dem, cmap='terrain', alpha=0.5)
    if show_rain and 'rainfall' in locals():
        ax.imshow(rain_norm, cmap='Blues', alpha=0.4)
    if show_geo and geology_raster is not None:
        # show geology classes with discrete colormap
        ax.imshow(geology_raster, alpha=0.35)
    if show_risk:
        heat = ax.imshow(combined_norm, cmap='hot', alpha=0.6)
        cbar = plt.colorbar(heat, ax=ax, fraction=0.036, pad=0.04)
        cbar.set_label("Risk (normalized)")

    ax.set_title("Layer Preview")
    ax.axis('off')
    st.pyplot(fig)

    # -------------------------
    # Download outputs
    # -------------------------
    st.subheader("Download Hasil")

    # 1) GeoTIFF export (risk map) using DEM profile
    out_profile = dem_meta.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    # write GeoTIFF to bytes
    tiff_bytes = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as out_tmp:
        out_path = out_tmp.name
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(combined_norm.astype(rasterio.float32), 1)
    with open(out_path, 'rb') as f:
        tiff_bytes = f.read()

    st.download_button("Download GeoTIFF (Risk Map)", data=tiff_bytes, file_name="risk_map.tif", mime="image/tiff")

    # 2) PNG heatmap
    png_buffer = io.BytesIO()
    fig2, ax2 = plt.subplots(figsize=(10,8))
    im = ax2.imshow(combined_norm, cmap='hot')
    ax2.axis('off')
    fig2.colorbar(im, ax=ax2, fraction=0.036, pad=0.04)
    plt.tight_layout()
    fig2.savefig(png_buffer, format='png', bbox_inches='tight', dpi=150)
    png_buffer.seek(0)
    st.download_button("Download PNG Heatmap", data=png_buffer, file_name="risk_map.png", mime="image/png")

    # 3) Vectorize risk map to polygons and export GeoJSON / Shapefile
    st.info("Membuat zonasi poligon dari raster risiko (thresholding optional untuk mengurangi banyak polygon).")
    # Ask thresholds to create classes
    bins = st.slider("Jumlah kelas zona (quantiles) untuk vectorize", min_value=2, max_value=6, value=4)
    # compute quantile bins
    flat = combined_norm.flatten()
    flat = flat[~np.isnan(flat)]
    thresholds = np.quantile(flat, np.linspace(0,1,bins+1))
    # create integer raster classes
    class_raster = np.full_like(combined_norm, 0, dtype=np.int16)
    for i in range(bins):
        lo = thresholds[i]
        hi = thresholds[i+1]
        mask = (combined_norm >= lo) & (combined_norm <= hi)
        class_raster[mask] = i+1

    # extract shapes
    shapes_gen = features.shapes(class_raster.astype(np.int16), mask=(class_raster>0), transform=dem_transform)
    geoms = []
    vals = []
    for geom, val in shapes_gen:
        geoms.append(shape(geom))
        vals.append(int(val))
    gdf = gpd.GeoDataFrame({"zone": vals}, geometry=geoms, crs=dem_crs)

    # add representative risk value for each polygon (mean)
    # compute mean from raster for each polygon
    # For performance we skip exact zonal stats; use simple approach:
    # (This can be improved using rasterstats package.)
    gdf["mean_risk"] = gdf["zone"].apply(lambda z: float(z) / bins)  # approximate label

    # export GeoJSON bytes
    geojson_bytes = gdf.to_json().encode('utf-8')
    st.download_button("Download GeoJSON (vector zones)", data=geojson_bytes, file_name="risk_zones.geojson", mime="application/geo+json")

    # export shapefile as zip
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = os.path.join(tmpdir, "risk_zones.shp")
        gdf.to_file(shp_path)
        # zip all files from tmpdir
        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, mode='w') as zf:
            for fname in os.listdir(tmpdir):
                zf.write(os.path.join(tmpdir, fname), arcname=fname)
        zip_io.seek(0)
        st.download_button("Download Shapefile (ZIP)", data=zip_io, file_name="risk_zones_shp.zip", mime="application/zip")

    st.success("Selesai — unduh file sesuai kebutuhan.")
else:
    st.info("Unggah DEM (.tif) untuk memulai proses.")
