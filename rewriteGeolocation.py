import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS


import xml.etree.ElementTree as ET

# Parse the XML metadata
tree = ET.parse('Data/Bering_WV/WV01_20110723210738_102001001591B200_11JUL23210738-P1BS-052792752100_01_P006.xml')
root = tree.getroot()
# Extract upper-left corner coordinates and pixel size
geo_transform = root.find('GeoTransform')

if geo_transform is not None:
    try:
        west = float(geo_transform.find('UpperLeftX').text)
    except AttributeError:
        raise ValueError("UpperLeftX not found in the XML file")

    try:
        north = float(geo_transform.find('UpperLeftY').text)
    except AttributeError:
        raise ValueError("UpperLeftY not found in the XML file")

    try:
        pixel_width = float(geo_transform.find('PixelWidth').text)
    except AttributeError:
        raise ValueError("PixelWidth not found in the XML file")

    try:
        pixel_height = float(geo_transform.find('PixelHeight').text)
    except AttributeError:
        raise ValueError("PixelHeight not found in the XML file")
    
    # Ensure pixel_height is negative
    if pixel_height > 0:
        pixel_height = -pixel_height

    # Create the transform
    transform = from_origin(west, north, pixel_width, pixel_height)
else:
    raise ValueError("GeoTransform section not found in the XML file")

# Ensure pixel_height is negative
if pixel_height > 0:
    pixel_height = -pixel_height

# Create the transform
transform = from_origin(west, north, pixel_width, pixel_height)

# Open the non-georeferenced TIFF
with rasterio.open('Data/Bering_WV/WV01_20110723210738_102001001591B200_11JUL23210738-P1BS-052792752100_01_P006.tif') as src:
    profile = src.profile

# Define the CRS and transform manually
crs = CRS.from_epsg(4326)  # or any other EPSG code
transform = from_origin(west, north, pixel_width, pixel_height)  # provide appropriate values

# Update the profile with the CRS and transform
profile.update({
    'crs': crs,
    'transform': transform
})

# Write the updated profile to a new file
with rasterio.open('Data/Bering_WV/WV01_06232011.tiff', 'w', **profile) as dst:
    dst.write(src.read())