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

# Load ephemeris and star data
eph = load('de421.bsp')
with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

# Settings
chart_size = 8
max_star_size = 80
limiting_magnitude = 6


def generate_star_chart(location_name, when_str, canvas_frame, draw_grid):
    try:
        locator = Nominatim(user_agent='star_chart_app')
        location = locator.geocode(location_name)
        if location is None:
            raise ValueError("Location not found.")
        lat, lon = location.latitude, location.longitude

        dt = datetime.strptime(when_str, '%Y-%m-%d %H:%M')
        timezone_str = tzwhere.tzwhere().tzNameAt(lat, lon)
        local = timezone(timezone_str)
        local_dt = local.localize(dt)
        utc_dt = local_dt.astimezone(utc)

        ts = load.timescale()
        t = ts.from_datetime(utc_dt)
        earth = eph['earth']
        observer = wgs84.latlon(lat, lon).at(t)

        ra, dec, _ = observer.radec()
        center_object = Star(ra=ra, dec=dec)
        center = earth.at(t).observe(center_object)
        projection = build_stereographic_projection(center)

        star_positions = earth.at(t).observe(Star.from_dataframe(stars))
        stars['x'], stars['y'] = projection(star_positions)

        bright_stars = stars['magnitude'] <= limiting_magnitude
        mag = stars['magnitude'][bright_stars]
        marker_size = max_star_size * 10 ** (mag / -2.5)

        fig, ax = plt.subplots(figsize=(chart_size, chart_size))
        ax.set_facecolor('black')
        border = Circle((0, 0), 1, color='navy', fill=True)
        ax.add_patch(border)

        ax.scatter(stars['x'][bright_stars], stars['y'][bright_stars],
                   s=marker_size, color='white', marker='.', linewidths=0, zorder=2)

        # Clip stars outside horizon
        horizon = Circle((0, 0), 1, transform=ax.transData)
        for col in ax.collections:
            col.set_clip_path(horizon)

        # Optional: Alt-Az grid lines
        if draw_grid:
            for alt_deg in range(15, 90, 15):
                r = np.tan(np.radians(90 - alt_deg) / 2)
                circle = plt.Circle((0, 0), r, edgecolor='gray', facecolor='none',
                                    linestyle='--', linewidth=0.5)
                ax.add_patch(circle)
                ax.text(0, r, f"{alt_deg}°", color='gray', fontsize=7,
                        ha='center', va='bottom')

            for az_deg in range(0, 360, 30):
                angle = np.radians(az_deg)
                x = np.cos(angle)
                y = np.sin(angle)
                ax.plot([0, x], [0, y], color='gray', linestyle='--', linewidth=0.5)
                ax.text(1.05 * x, 1.05 * y, f"{az_deg}°", color='gray', fontsize=7,
                        ha='center', va='center')

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')

        for widget in canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        plt.close(fig)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# GUI setup
root = tk.Tk()
root.title("StarScope Viewer")

frame_inputs = tk.Frame(root)
frame_inputs.pack(pady=5)

tk.Label(frame_inputs, text="Location (e.g., 'Boston, MA')").grid(row=0, column=0, sticky='w')
location_entry = tk.Entry(frame_inputs, width=35)
location_entry.grid(row=1, column=0, padx=5, pady=2)

tk.Label(frame_inputs, text="Date & Time (YYYY-MM-DD HH:MM)").grid(row=2, column=0, sticky='w')
time_entry = tk.Entry(frame_inputs, width=25)
time_entry.insert(0, "2023-01-01 00:00")
time_entry.grid(row=3, column=0, padx=5, pady=2)

grid_var = tk.IntVar()
grid_checkbox = tk.Checkbutton(frame_inputs, text="Show Alt-Az Grid Lines", variable=grid_var)
grid_checkbox.grid(row=4, column=0, sticky='w', pady=5)

generate_button = tk.Button(frame_inputs, text="Generate Star Chart",
                            command=lambda: generate_star_chart(
                                location_entry.get(),
                                time_entry.get(),
                                canvas_frame,
                                grid_var.get() == 1))
generate_button.grid(row=5, column=0, pady=10)

canvas_frame = tk.Frame(root)
canvas_frame.pack(padx=10, pady=10)

root.mainloop()
