import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_driving_report():
    """
    Creates a comprehensive driving report from GPS tracking data
    Saves as 'driving_report.txt' and creates visualizations
    """
    
    # Load the CSV data
    print("Loading driving data...")
    df = pd.read_csv('trackLog-2025-Oct-08_13-37-08.csv', low_memory=False)
    
    # Find relevant columns
    gps_lat_col = None
    gps_lon_col = None
    speed_col = None
    time_col = None
    
    for col in df.columns:
        if 'lat' in col.lower() and 'gps' in col.lower() and 'bearing' not in col.lower():
            gps_lat_col = col
        if 'lon' in col.lower() and 'gps' in col.lower() and 'bearing' not in col.lower():
            gps_lon_col = col
        if 'speed' in col.lower() and 'gps' in col.lower():
            speed_col = col
        if 'time' in col.lower() and 'device' in col.lower():
            time_col = col
    
    print(f"Found columns: Lat={gps_lat_col}, Lon={gps_lon_col}, Speed={speed_col}")
    
    # Clean data
    df = df.dropna(subset=[gps_lat_col, gps_lon_col, speed_col])
    df = df[(df[gps_lat_col] != 0) & (df[gps_lon_col] != 0)]
    df[speed_col] = pd.to_numeric(df[speed_col], errors='coerce')
    df = df.dropna(subset=[speed_col])
    
    print(f"Data points after cleaning: {len(df)}")
    
    # Calculate driving metrics
    speeds = df[speed_col]
    avg_speed = speeds.mean()
    max_speed = speeds.max()
    min_speed = speeds.min()
    median_speed = speeds.median()
    
    # Speed analysis
    speed_changes = speeds.diff().dropna()
    hard_braking_events = (speed_changes < -5).sum()  # sudden deceleration > 5 km/h
    hard_acceleration_events = (speed_changes > 8).sum()  # sudden acceleration > 8 km/h
    
    # Distance calculation (approximate)
    lat_diff = df[gps_lat_col].diff() * 111.32  # km per degree latitude
    lon_diff = df[gps_lon_col].diff() * 111.32 * np.cos(np.radians(df[gps_lat_col]))
    distances = np.sqrt(lat_diff**2 + lon_diff**2)
    total_distance = distances.sum()
    
    # Time analysis
    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            trip_duration = (df[time_col].max() - df[time_col].min()).total_seconds() / 60  # minutes
        except:
            trip_duration = len(df)  # fallback to data points
    else:
        trip_duration = len(df)  # fallback
    
    # Driving behavior analysis
    speeding_threshold = 60  # km/h (adjust as needed)
    speeding_events = (speeds > speeding_threshold).sum()
    
    idle_time = (speeds == 0).sum()
    moving_time = len(df) - idle_time
    
    # Speed categories
    low_speed = (speeds <= 20).sum()
    medium_speed = ((speeds > 20) & (speeds <= 50)).sum()
    high_speed = (speeds > 50).sum()
    
    # Create the report content
    report_content = f"""
DRIVING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================

TRIP OVERVIEW
-------------
• Total Distance: {total_distance:.2f} km
• Trip Duration: {trip_duration:.1f} minutes
• Total Data Points: {len(df)}

SPEED ANALYSIS
--------------
• Average Speed: {avg_speed:.1f} km/h
• Maximum Speed: {max_speed:.1f} km/h
• Minimum Speed: {min_speed:.1f} km/h
• Median Speed: {median_speed:.1f} km/h

DRIVING BEHAVIOR
----------------
• Hard Braking Events: {hard_braking_events}
• Hard Acceleration Events: {hard_acceleration_events}
• Speeding Events (>{speeding_threshold} km/h): {speeding_events}
• Time Spent Idle: {idle_time} data points
• Time Spent Moving: {moving_time} data points

SPEED DISTRIBUTION
------------------
• Low Speed (0-20 km/h): {low_speed} points ({low_speed/len(df)*100:.1f}%)
• Medium Speed (21-50 km/h): {medium_speed} points ({medium_speed/len(df)*100:.1f}%)
• High Speed (>50 km/h): {high_speed} points ({high_speed/len(df)*100:.1f}%)

DRIVING SCORE
-------------
"""
    
    # Calculate driving score
    score = 100
    score -= hard_braking_events * 2  # -2 points per hard brake
    score -= hard_acceleration_events * 1.5  # -1.5 points per hard acceleration
    score -= speeding_events * 0.5  # -0.5 points per speeding event
    score = max(0, min(100, score))  # Keep between 0-100
    
    if score >= 90:
        grade = "Excellent"
    elif score >= 80:
        grade = "Good"
    elif score >= 70:
        grade = "Fair"
    elif score >= 60:
        grade = "Needs Improvement"
    else:
        grade = "Poor"
    
    report_content += f"Overall Driving Score: {score:.1f}/100 ({grade})\n\n"
    
    # Recommendations
    report_content += "RECOMMENDATIONS\n"
    report_content += "---------------\n"
    if hard_braking_events > 5:
        report_content += "• Try to anticipate traffic and brake more gradually\n"
    if hard_acceleration_events > 8:
        report_content += "• Accelerate more smoothly to save fuel and reduce wear\n"
    if speeding_events > 10:
        report_content += "• Monitor speed limits more carefully\n"
    if avg_speed < 10:
        report_content += "• Consider alternate routes to avoid heavy traffic\n"
    if score >= 85:
        report_content += "• Great driving! Keep up the safe driving habits\n"
    
    report_content += "\nRoute coordinates saved for mapping purposes.\n"
    report_content += f"GPS Data Range: {df[gps_lat_col].min():.6f} to {df[gps_lat_col].max():.6f} (Lat)\n"
    report_content += f"                {df[gps_lon_col].min():.6f} to {df[gps_lon_col].max():.6f} (Lon)\n"
    
    # Save the report
    with open('driving_report.txt', 'w') as f:
        f.write(report_content)
    
    # Create route visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Route map
    plt.subplot(2, 2, 1)
    plt.plot(df[gps_lon_col], df[gps_lat_col], 'b-', alpha=0.7, linewidth=1)
    plt.scatter(df[gps_lon_col].iloc[0], df[gps_lat_col].iloc[0], color='green', s=100, label='Start', zorder=5)
    plt.scatter(df[gps_lon_col].iloc[-1], df[gps_lat_col].iloc[-1], color='red', s=100, label='End', zorder=5)
    plt.title('Driving Route')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Speed over time
    plt.subplot(2, 2, 2)
    plt.plot(speeds.values, 'b-', alpha=0.7)
    plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Avg: {avg_speed:.1f} km/h')
    plt.title('Speed Over Time')
    plt.xlabel('Data Point')
    plt.ylabel('Speed (km/h)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Speed distribution
    plt.subplot(2, 2, 3)
    plt.hist(speeds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=avg_speed, color='r', linestyle='--', label=f'Average: {avg_speed:.1f}')
    plt.title('Speed Distribution')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Driving behavior summary
    plt.subplot(2, 2, 4)
    categories = ['Low\n(0-20)', 'Medium\n(21-50)', 'High\n(>50)']
    values = [low_speed, medium_speed, high_speed]
    colors = ['lightgreen', 'orange', 'red']
    plt.bar(categories, values, color=colors, alpha=0.7)
    plt.title('Speed Categories')
    plt.ylabel('Data Points')
    for i, v in enumerate(values):
        plt.text(i, v + max(values)*0.01, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('driving_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Driving report saved as 'driving_report.txt'")
    print("✓ Visualizations saved as 'driving_analysis.png'")
    print(f"✓ Analysis complete! Your driving score: {score:.1f}/100 ({grade})")
    
    return report_content

# Run the analysis
if __name__ == "__main__":
    create_driving_report()
