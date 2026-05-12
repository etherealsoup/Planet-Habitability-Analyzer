import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import tkinter as tk
from tkinter import messagebox

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv('exo.csv')  # make sure file is in same folder

print("Class distribution:")
print(df['habitable'].value_counts())

# ===============================
# PREPARE DATA
# ===============================
X = df[["distance", "temperature", "radius"]].values
y = df["habitable"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# MODEL
# ===============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ===============================
# GRAPH FUNCTION (SIMPLE + CLEAN)
# ===============================
def show_graphs():

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    # 1 Histogram
    axs[0, 0].hist(df['temperature'])
    axs[0, 0].set_title("Temperature Distribution")

    # 2 Boxplot
    axs[0, 1].boxplot(df['radius'])
    axs[0, 1].set_title("Radius Spread")

    # 3 Bar chart
    df['habitable'].value_counts().plot(
        kind='bar',
        ax=axs[0, 2]
    )
    axs[0, 2].set_title("Habitable vs Not")

    # 4 Scatter plot
    colors = []

    for i in range(len(df)):
        temp = df['temperature'].iloc[i]
        dist = df['distance'].iloc[i]

        if 0.7 < dist < 2.0 and 220 < temp < 330:
            colors.append('green')
        elif temp >= 330:
            colors.append('red')
        else:
            colors.append('blue')

    axs[1, 0].scatter(df['distance'], df['temperature'], c=colors)
    axs[1, 0].set_title("Habitability Zones")
    axs[1, 0].set_xlabel("Distance (AU)")
    axs[1, 0].set_ylabel("Temperature (K)")

    # 5 Heatmap
    sns.heatmap(df.corr(), annot=True, ax=axs[1, 1])
    axs[1, 1].set_title("Feature Correlation")

    # Empty last box
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

# ===============================
# OPENCV PLANET VISUAL
# ===============================
def show_planet():
    try:
        d = float(entry_distance.get())
        t = float(entry_temperature.get())
        r = float(entry_radius.get())
    except:
        messagebox.showerror("Error", "Enter valid inputs first")
        return

    # Create black space background
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Decide color based on temperature
    if t > 330:
        color = (0, 0, 255)   # red (BGR)
    elif t < 220:
        color = (255, 0, 0)   # blue
    else:
        color = (0, 255, 0)   # green

    # Scale radius (make it visible)
    planet_size = int(30 + r * 40)

    # Draw planet
    cv2.circle(img, (200, 200), planet_size, color, -1)

    # Add label text
    label = "Habitable Zone" if (0.7 < d < 2.0 and 220 < t < 330) else "Extreme Conditions"
    cv2.putText(img, label, (80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show window
    cv2.imshow("Planet Simulation", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_habitability(distance, temperature, radius):
    features = np.array([[distance, temperature, radius]])
    features_scaled = scaler.transform(features)

    ideal_distance = 1.2
    ideal_temp = 290
    ideal_radius = 1.0

    distance_score = max(0, 1 - abs(distance - ideal_distance)/5)
    temp_score = max(0, 1 - abs(temperature - ideal_temp)/200)
    radius_score = max(0, 1 - abs(radius - ideal_radius)/2)

    prob = (distance_score + temp_score + radius_score) / 3

    reasons = []
    suggestions = []

    # Ideal values
    ideal_distance = 1.2
    ideal_temp = 290
    ideal_radius = 1.0

    # Temperature checks
    if temperature < 220:
        reasons.append("Too Cold ❄️")
        suggestions.append(f"Increase temperature toward {ideal_temp} K")
    elif temperature > 330:
        reasons.append("Too Hot 🔥")
        suggestions.append(f"Reduce temperature toward {ideal_temp} K")

    # Distance checks
    if distance < 0.7:
        reasons.append("Too Close to Star ☀️")
        suggestions.append(f"Increase distance toward {ideal_distance} AU")
    elif distance > 2.0:
        reasons.append("Too Far from Star 🌌")
        suggestions.append(f"Reduce distance toward {ideal_distance} AU")

    # Radius checks
    if radius < 0.7:
        reasons.append("Too Small 🪨")
        suggestions.append(f"Increase radius toward {ideal_radius}")
    elif radius > 2.0:
        reasons.append("Too Big 🌍")
        suggestions.append(f"Reduce radius toward {ideal_radius}")

    # If no strong problems but confidence < 90%
    if prob < 0.90 and not suggestions:

        if distance != ideal_distance:
            if distance < ideal_distance:
                suggestions.append(
                    f"Increase distance toward {ideal_distance} AU"
                )
            else:
                suggestions.append(
                    f"Reduce distance toward {ideal_distance} AU"
                )

        if temperature != ideal_temp:
            if temperature < ideal_temp:
                suggestions.append(
                    f"Increase temperature toward {ideal_temp} K"
                )
            else:
                suggestions.append(
                    f"Reduce temperature toward {ideal_temp} K"
                )

        if radius != ideal_radius:
            if radius < ideal_radius:
                suggestions.append(
                    f"Increase radius toward {ideal_radius}"
                )
            else:
                suggestions.append(
                    f"Reduce radius toward {ideal_radius}"
                )

    # Final output
    if prob >= 0.90:
        return (
            f"Highly Habitable 🌱\n"
            f"Confidence: {prob*100:.2f}%"
        )

    elif prob >= 0.50:
        reason_text = ", ".join(reasons) if reasons else "Conditions can be optimized"
        suggestion_text = "\n".join(suggestions)

        return (
            f"Potentially Habitable 🌍\n"
            f"Confidence: {prob*100:.2f}%\n"
            f"Status: {reason_text}\n\n"
            f"Suggested Improvements:\n{suggestion_text}"
        )

    else:
        reason_text = ", ".join(reasons) if reasons else "Not suitable for life"
        suggestion_text = "\n".join(suggestions)

        return (
            f"Not Habitable ❌\n"
            f"Confidence: {prob*100:.2f}%\n"
            f"Status: {reason_text}\n\n"
            f"Suggested Improvements:\n{suggestion_text}"
        )

# ===============================
# UI
# ===============================
root = tk.Tk()
root.title("AI Exoplanet Habitability Analyzer")
root.geometry("500x550")
root.configure(bg="#0b1020")

# Title
title_label = tk.Label(
    root,
    text="🌌 Exoplanet Habitability Analyzer",
    font=("Arial", 18, "bold"),
    bg="#0b1020",
    fg="white"
)
title_label.pack(pady=15)

# Main Frame
main_frame = tk.Frame(root, bg="#111827", padx=20, pady=20)
main_frame.pack(pady=10, padx=20, fill="both", expand=True)

# Input Fields + Sliders
input_frame = tk.Frame(main_frame, bg="#111827")
input_frame.pack(pady=10)

def sync_slider(entry, slider, min_val, max_val):
    try:
        value = float(entry.get())

        # Clamp values (prevents weird behavior)
        if value < min_val:
            value = min_val
        elif value > max_val:
            value = max_val

        slider.set(value)

    except:
        pass


def make_input_with_slider(label_text, min_val, max_val):

    row_frame = tk.Frame(input_frame, bg="#111827")
    row_frame.pack(fill="x", pady=8)

    # Label (left)
    tk.Label(
        row_frame,
        text=label_text,
        font=("Arial", 11, "bold"),
        bg="#111827",
        fg="white",
        width=15,
        anchor="w"
    ).pack(side="left")

    # Textbox (left-middle)
    entry = tk.Entry(
        row_frame,
        font=("Arial", 12),
        width=10
    )
    entry.pack(side="left", padx=10)

    # Slider (center/right)
    slider = tk.Scale(
        row_frame,
        from_=min_val,
        to=max_val,
        orient="horizontal",
        resolution=0.01,
        length=220,
        showvalue=True,   # always shows current value
        bg="#111827",
        fg="white",
        troughcolor="#374151",
        highlightthickness=0
    )
    slider.pack(side="left", padx=10)

    # Slider → textbox
    def slider_changed(value):
        entry.delete(0, tk.END)
        entry.insert(0, str(round(float(value), 2)))

    slider.config(command=slider_changed)

    # Textbox → slider
    # Update slider only when user finishes typing
    entry.bind(
        "<FocusOut>",
        lambda event: sync_slider(entry, slider, min_val, max_val)
    )

    entry.bind(
        "<Return>",
        lambda event: sync_slider(entry, slider, min_val, max_val)
    )

    # Set default to middle
    default_value = (min_val + max_val) / 2
    slider.set(default_value)
    entry.insert(0, str(round(default_value, 2)))

    return entry, slider


entry_distance, distance_slider = make_input_with_slider(
    "Distance (AU)",
    float(df['distance'].min()),
    float(df['distance'].max())
)

entry_temperature, temperature_slider = make_input_with_slider(
    "Temperature (K)",
    float(df['temperature'].min()),
    float(df['temperature'].max())
)

entry_radius, radius_slider = make_input_with_slider(
    "Radius",
    float(df['radius'].min()),
    float(df['radius'].max())
)

# Result Label
result_label = tk.Label(
    main_frame,
    text="Enter planet data to analyze...",
    font=("Arial", 11),
    bg="#111827",
    fg="#34D399",
    wraplength=350
)
result_label.pack(pady=20)

# Predict button
def predict_from_ui():
    try:
        d = float(entry_distance.get())
        t = float(entry_temperature.get())
        r = float(entry_radius.get())

        result = predict_habitability(d, t, r)
        result_label.config(text=result)

    except:
        messagebox.showerror("Error", "Enter valid numbers")

# Button style
button_style = {
    "font": ("Arial", 11, "bold"),
    "width": 25,
    "height": 2,
    "bd": 0
}

tk.Button(
    main_frame,
    text="🔍 Predict Habitability",
    command=predict_from_ui,
    bg="#2563EB",
    fg="white",
    **button_style
).pack(pady=8)

tk.Button(
    main_frame,
    text="📊 Show Graphs",
    command=show_graphs,
    bg="#7C3AED",
    fg="white",
    **button_style
).pack(pady=8)

tk.Button(
    main_frame,
    text="🪐 Show Planet Visual",
    command=show_planet,
    bg="#059669",
    fg="white",
    **button_style
).pack(pady=8)

root.mainloop()
