import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow/MediaPipe logs
import sys

# Import each exercise module (assumes they're in the same directory or a package)
import lunges
import squats
import plank
import pushups
import sidelyinglegraises
import yoga_pose_classifier

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_yes_or_no(prompt):
    while True:
        choice = input(f"{prompt} (y/n): ").strip().lower()
        if choice in ['y', 'n']:
            return choice == 'y'
        print("Invalid input. Please enter 'y' or 'n'.")

def get_positive_number(prompt):
    while True:
        try:
            val = float(input(prompt))
            if val > 0:
                return val
            print("Enter a positive number.")
        except:
            print("Invalid input. Enter a number.")

def main():
    clear()
    print("\n👋 Welcome to the Smart Fitness Tracker!")

    # Video source setup
    print("\nWould you like to use sample videos or your webcam?")
    use_sample_videos = get_yes_or_no("Use sample videos?")

    sample_videos = {
        'plank': "Sample Videos/planksample.mp4",
        'lunges': "Sample Videos/_plunges.mp4",
        'pushups': "Sample Videos/pushups.mp4",
        'sidelyinglegraises': "Sample Videos/sidelyinglegraise.mp4",
        'squats': "Sample Videos/squatssample.mp4"
    }

    user_weight = None

    while True:
        print("\nWhich workout would you like to perform?")
        print("1. Lunges")
        print("2. Pushups")
        print("3. Squats")
        print("4. Side-Lying Leg Raises")
        print("5. Plank")
        print("6. Yoga")

        choice = input("Enter your choice (1-6): ").strip()

        if user_weight is None:
            user_weight = get_positive_number("Enter your weight (in kg): ")

        video_path = None
        if use_sample_videos:
            # Use corresponding sample
            video_path = sample_videos.get(
                'plank' if choice == '5' else
                'lunges' if choice == '1' else
                'pushups' if choice == '2' else
                'sidelyinglegraises' if choice == '4' else
                'squats' if choice == '3' else None
            )
        else:
            video_path = 3  # Webcam
        print(f"Using video path: {video_path}")
        result = {}

        if choice == '1':  # Lunges
            target_reps = int(get_positive_number("Enter target number of reps: "))
            result = lunges.run_lunges(user_weight, target_reps, video_path)

        elif choice == '2':  # Pushups
            target_reps = int(get_positive_number("Enter target number of reps: "))
            result = pushups.run_pushups(user_weight, target_reps, video_path)

        elif choice == '3':  # Squats
            target_reps = int(get_positive_number("Enter target number of reps: "))
            result = squats.run_squats(user_weight, target_reps, video_path)

        elif choice == '4':  # Side-Lying Leg Raises
            target_reps = int(get_positive_number("Enter target number of reps: "))
            result = sidelyinglegraises.run_sidelying_leg_raises(user_weight, target_reps, video_path)

        elif choice == '5':  # Plank
            target_time = int(get_positive_number("Enter target hold time (in seconds): "))
            result = plank.run_plank(user_weight, target_time, video_path)

        elif choice == '6':  # Yoga
            print("\nSelect Yoga Pose:")
            print("  a. T Pose")
            print("  b. Warrior II Pose")
            print("  c. Chair Pose")
            yoga_choice = input("Enter your choice (a/b/c): ").strip().lower()

            pose_name = {
                'a': 'T Pose',
                'b': 'Warrior II Pose',
                'c': 'Chair Pose'
            }.get(yoga_choice)

            if pose_name is None:
                print("Invalid yoga pose selected. Returning to main menu.")
                continue

            target_time = int(get_positive_number("Enter target hold time (in seconds): "))
            print("Function called")
            result = yoga_pose_classifier.run_yoga_pose(user_weight, target_time, pose_name, video_path)

        else:
            print("\nInvalid choice. Please try again.")
            continue

        print("\n----- Workout Result -----")
        print(f"Status     : {result.get('status', 'unknown')}")
        print(f"Calories   : {result.get('calories', 0):.2f} kcal")
        if 'reps_completed' in result:
            print(f"Reps Done  : {result['reps_completed']}")
        if 'time_held' in result:
            print(f"Time Held  : {result['time_held']:.2f} seconds")
        
        if result.get('status') == 'Success':
            print("✅ Target achieved! Well done!")
        else:
            print("❌ Target not met. Keep practicing!")

        again = get_yes_or_no("\nDo you want to perform another exercise?")
        if not again:
            print("\nThanks for using the Smart Fitness Tracker. Stay healthy! 💪")
            break

if __name__ == '__main__':
    main()
