fp = r'Resources_Organized/tif/Bestand3.tif'
bestand_tif = rasterio.open(fp)
show(bestand_tif, aspect='auto')