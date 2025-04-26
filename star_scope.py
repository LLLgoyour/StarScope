import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from geopy.geocoders import Nominatim
from tzwhere import tzwhere
from pytz import timezone, utc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle

from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos
from skyfield.projections import build_stereographic_projection

# Settings
chart_size = 8            # Size of the star chart in inches
max_star_size = 80        # Maximum marker size for the brightest stars
limiting_magnitude = 6    # Only plot stars brighter than this magnitude

# Load ephemeris and star catalog
eph = load('de421.bsp')
with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

# Pre-instantiate tzwhere for performance
tz_finder = tzwhere.tzwhere()

def generate_star_chart(location_name, when_str, canvas_frame, draw_grid):
    try:
        # Geocode the location
        locator = Nominatim(user_agent='star_chart_app',timeout=10)
        loc = locator.geocode(location_name)
        if loc is None:
            raise ValueError(f"Location '{location_name}' not found.")
        lat, lon = loc.latitude, loc.longitude

        # Parse local date-time
        dt = datetime.strptime(when_str, '%Y-%m-%d %H:%M')
        tz_name = tz_finder.tzNameAt(lat, lon)
        if tz_name is None:
            raise ValueError("Cannot determine timezone for this location.")
        local_tz = timezone(tz_name)
        local_dt = local_tz.localize(dt)
        utc_dt = local_dt.astimezone(utc)

        # Compute sky positions
        ts = load.timescale()
        t = ts.from_datetime(utc_dt)
        earth = eph['earth']
        observer = wgs84.latlon(lat, lon).at(t)

        # Build stereographic projection centered on zenith
        ra, dec, _ = observer.radec()
        center_star = Star(ra=ra, dec=dec)
        center_obs = earth.at(t).observe(center_star)
        proj = build_stereographic_projection(center_obs)

        # Project all stars and filter by magnitude
        star_obs = earth.at(t).observe(Star.from_dataframe(stars))
        stars['x'], stars['y'] = proj(star_obs)
        bright = stars['magnitude'] <= limiting_magnitude
        df = stars[bright].copy().reset_index()  # 'index' holds HIP ID
        mags = df['magnitude']
        sizes = max_star_size * 10 ** (mags / -2.5)

        # Create figure
        fig, ax = plt.subplots(figsize=(chart_size, chart_size))
        ax.set_facecolor('black')
        ax.add_patch(Circle((0, 0), 1, color='navy', fill=True))

        # Scatter plot of stars
        scatter = ax.scatter(df['x'], df['y'], s=sizes,
                             color='white', marker='.', zorder=2)
        horizon = Circle((0, 0), 1, transform=ax.transData)
        for col in ax.collections:
            col.set_clip_path(horizon)

        # Optional Alt-Az grid
        if draw_grid:
            for alt in range(15, 90, 15):
                r = np.tan(np.radians(90 - alt) / 2)
                circ = plt.Circle((0, 0), r, edgecolor='gray',
                                  facecolor='none', linestyle='--', linewidth=0.5)
                ax.add_patch(circ)
                ax.text(0, r, f"{alt}°", color='gray', fontsize=7,
                        ha='center', va='bottom')
            for az in range(0, 360, 30):
                ang = np.radians(az)
                x, y = np.cos(ang), np.sin(ang)
                ax.plot([0, x], [0, y], color='gray',
                        linestyle='--', linewidth=0.5)
                ax.text(1.05 * x, 1.05 * y, f"{az}°", color='gray',
                        fontsize=7, ha='center', va='center')

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')

        # Interactive annotation
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.8),
            arrowprops=dict(arrowstyle="->")
        )
        annot.set_visible(False)

        def update_annot(ind):
            idx = ind["ind"][0]
            x, y = scatter.get_offsets()[idx]
            annot.xy = (x, y)
            row = df.iloc[idx]
            hip_id = row['index']
            name = row.get('proper', '') if row.get('proper', '') else f"HIP {hip_id}"
            annot.set_text(f"{name}\nMag: {row['magnitude']:.2f}")

        def hover(event):
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', hover)

        # Embed in Tkinter
        for w in canvas_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        plt.close(fig)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# optional cache for geocoding
CACHE = {
    "Boston, MA": (42.3601, -71.0589),
    # ... other locations
}

# GUI Setup
root = tk.Tk()
root.title("StarScope Viewer")

frame_inputs = tk.Frame(root)
frame_inputs.pack(pady=5)

tk.Label(frame_inputs, text="Location (e.g., 'Boston, MA')").grid(row=0, column=0, sticky='w')
location_entry = tk.Entry(frame_inputs, width=35)
location_entry.grid(row=1, column=0, padx=5, pady=2)

tk.Label(frame_inputs, text="Date & Time (YYYY-MM-DD HH:MM)").grid(row=2, column=0, sticky='w')
time_entry = tk.Entry(frame_inputs, width=25)
time_entry.insert(0, datetime.utcnow().strftime('%Y-%m-%d %H:%M'))
time_entry.grid(row=3, column=0, padx=5, pady=2)

grid_var = tk.IntVar()
grid_checkbox = tk.Checkbutton(frame_inputs, text="Show Alt-Az Grid Lines", variable=grid_var)
grid_checkbox.grid(row=4, column=0, sticky='w', pady=5)

generate_button = tk.Button(
    frame_inputs, text="Generate Star Chart",
    command=lambda: generate_star_chart(
        location_entry.get(), time_entry.get(), canvas_frame, grid_var.get() == 1)
)
generate_button.grid(row=5, column=0, pady=10)

canvas_frame = tk.Frame(root)
canvas_frame.pack(padx=10, pady=10)

root.mainloop()