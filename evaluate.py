import xml.etree.ElementTree as ET
import numpy as np

def get_average_travel_time(file_path):
    """
    Parses a SUMO tripinfo XML file and returns the average travel time.

    Args:
        file_path (str): The path to the tripinfo.xml file.

    Returns:
        float: The average travel time of all vehicles, or 0 if the file is not found.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        travel_times = []
        for tripinfo in root.findall('tripinfo'):
            duration = tripinfo.get('duration')
            if duration is not None:
                travel_times.append(float(duration))

        if not travel_times:
            print(f"Warning: No trip durations found in {file_path}")
            return 0.0

        return np.mean(travel_times)

    except FileNotFoundError:
        print(f"Error: Trip info file not found at {file_path}")
        return 0.0
    except ET.ParseError:
        print(f"Error: Could not parse XML file at {file_path}")
        return 0.0

if __name__ == "__main__":
    # File paths for the tripinfo outputs
    agent_trip_info_file = 'tripinfo.xml'
    fixed_timer_trip_info_file = 'tripinfo_fixed.xml'

    print("--- Traffic Management System Evaluation ---")

    # Calculate average travel times
    avg_time_fixed = get_average_travel_time(fixed_timer_trip_info_file)
    avg_time_agent = get_average_travel_time(agent_trip_info_file)

    if avg_time_fixed > 0 and avg_time_agent > 0:
        print(f"\nBaseline (Fixed Timer) Average Commute Time: {avg_time_fixed:.2f} seconds")
        print(f"RL Agent Average Commute Time: {avg_time_agent:.2f} seconds")

        # Calculate and print the improvement
        improvement = ((avg_time_fixed - avg_time_agent) / avg_time_fixed) * 100

        print(f"\nImprovement: {improvement:.2f}%")

        if improvement >= 10:
            print("\nCongratulations! The RL agent achieved the target 10% reduction in commute time.")
        else:
            print("\nThe RL agent did not meet the target 10% reduction. Further training or tuning is needed.")
    else:
        print("\nCould not calculate improvement due to missing or invalid data in one or both tripinfo files.")

    print("\n--- Evaluation Complete ---")
