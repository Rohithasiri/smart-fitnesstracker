import streamlit as st

# Import workout modules
from lunges import run_lunges
from pushups import run_pushups
from squats import run_squats
from sidelyinglegraises import run_sidelying_leg_raises
from plank import run_plank
from Crunches import run_crunches
from yoga_pose_classifier import run_yoga_pose

SCHEDULES = {
    "Beginner": {
        "Monday": ["Squats – 3 sets of 15 reps", "Pushups – 3 sets of 10 reps", "Tree Pose – 30 sec"],
        "Tuesday": ["Crunches – 3 sets of 15 reps", "Plank – 30 sec", "Warrior Pose – 30 sec"],
        "Wednesday": ["Lunges – 3 sets of 10 reps", "Side-lying leg raises – 3 sets of 12 reps", "Chair Pose – 30 sec"],
        "Thursday": ["Rest or gentle Yoga"],
        "Friday": ["Squats – 4 sets of 15 reps", "Pushups – 3 sets of 12 reps", "Tree Pose – 30 sec"],
        "Saturday": ["Crunches – 4 sets of 20 reps", "Plank – 45 sec", "Warrior Pose – 30 sec"],
        "Sunday": ["Rest or gentle Yoga"]
    },
     "Intermediate": {
        "Monday": ["Squats – 4 sets of 20 reps", "Pushups – 4 sets of 15 reps", "Tree Pose"],
        "Tuesday": ["Crunches – 4 sets of 20 reps", "Plank – 45 sec", "Warrior Pose"],
        "Wednesday": ["Lunges – 4 sets of 15 reps", "Side-lying leg raises – 4 sets of 15 reps", "Chair Pose"],
        "Thursday": ["Yoga Flow: Tree → Warrior → Chair Pose (hold 1 min each)"],
        "Friday": ["Squats – 5 sets of 20 reps", "Pushups – 4 sets of 20 reps", "Plank – 1 min"],
        "Saturday": ["Crunches – 5 sets of 25 reps", "Plank – 1 min", "Warrior Pose"],
        "Sunday": ["Rest or full-body Yoga flow"]
    },
    "Advanced": {
        "Monday": ["Squats – 5 sets of 25 reps", "Pushups – 5 sets of 20 reps", "Plank – 1.5 min"],
        "Tuesday": ["Crunches – 5 sets of 30 reps", "Lunges – 5 sets of 20 reps", "Chair Pose"],
        "Wednesday": ["Side-lying leg raises – 5 sets of 20 reps", "Warrior Pose – 1.5 min hold", "Tree Pose – 1.5 min hold"],
        "Thursday": ["Pushups – 6 sets of 25 reps", "Squats – 5 sets of 30 reps", "Plank – 2 min"],
        "Friday": ["Crunches – 5 sets of 35 reps", "Lunges – 6 sets of 25 reps", "Warrior Pose"],
        "Saturday": ["Chair Pose – 2 min hold", "Tree Pose – 2 min hold", "Plank – 2 min"],
        "Sunday": ["Rest or advanced Yoga flow"]
    }
   
}

def extract_target(target_str):
    nums = re.findall(r'\d+', target_str)
    if nums:
        return int(nums[0])
    return 0

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'schedule_queue' not in st.session_state:
    st.session_state.schedule_queue = []
if 'workout_results' not in st.session_state:
    st.session_state.workout_results = []
if 'weight' not in st.session_state:
    st.session_state.weight = None
if 'workout_name' not in st.session_state:
    st.session_state.workout_name = ''
if 'pose_name' not in st.session_state:
    st.session_state.pose_name = ''
if 'video_path' not in st.session_state:
    st.session_state.video_path = 0
if 'target_reps_or_time' not in st.session_state:
    st.session_state.target_reps_or_time = 0
if 'calories_per_workout' not in st.session_state:
    st.session_state.calories_per_workout = []
if 'total_calories' not in st.session_state:
    st.session_state.total_calories = 0
if 'in_schedule_mode' not in st.session_state:
    st.session_state.in_schedule_mode = False

def go_to(page):
    st.session_state.page = page
    st.rerun()

def is_hold_time_workout(name):
    name_lower = name.lower()
    return "plank" in name_lower or "pose" in name_lower or "yoga" in name_lower or "tree" in name_lower

# ---- Pages ----

if st.session_state.page == 'home':
    st.title("GetUpGo!")
    if st.session_state.weight is None:
        weight_val = st.number_input("Enter your weight (kg):", min_value=50, max_value=200)
        if st.button("Continue"):
            st.session_state.weight = weight_val
            go_to('menu')
    else:
        st.write(f"Current weight: **{st.session_state.weight} kg**")
        if st.button("Let's Begin"):
            go_to('menu')

elif st.session_state.page == 'menu':
    st.header("Main Menu")
    if st.button("Run a Workout (direct)"):
        st.session_state.in_schedule_mode = False
        go_to('workout_selection')
    if st.button("Schedule"):
        st.session_state.in_schedule_mode = True
        go_to('schedule')

elif st.session_state.page == 'workout_selection':
    st.header("Select a Workout")
    workouts = ["Lunges", "Pushups", "Squats", "Side-lying Leg Raises", "Plank", "Crunches", "Yoga"]
    choice = st.selectbox("Choose workout", workouts)
    
    if choice == "Yoga":
        yoga_postures = ["Tree Pose", "Chair Pose", "Warrior Pose"]
        posture_choice = st.selectbox("Select Yoga Posture", yoga_postures)
        st.session_state.workout_name = posture_choice
    else:
        st.session_state.workout_name = choice

    if st.button("Next"):
        go_to('set_target')

elif st.session_state.page == 'set_target':
    st.header(f"Set target for {st.session_state.workout_name}")
    if is_hold_time_workout(st.session_state.workout_name):
        target = st.number_input("Enter hold time (seconds):", min_value=1, max_value=600, value=30)
    else:
        target = st.number_input("Enter number of reps:", min_value=1, max_value=1000, value=10)

    if st.button("Start Workout"):
        st.session_state.target_reps_or_time = target
        go_to('workout')

elif st.session_state.page == 'schedule':
    st.header("Schedules")
    level = st.selectbox("Select Level", list(SCHEDULES.keys()))
    day = st.selectbox("Select Day", list(SCHEDULES[level].keys()))
    if st.button("Start Today's Schedule"):
        st.session_state.schedule_queue = SCHEDULES[level][day][:]
        st.session_state.calories_per_workout = []
        st.session_state.total_calories = 0
        go_to('run_schedule')

elif st.session_state.page == 'run_schedule':
    if not st.session_state.schedule_queue:
        st.success("Schedule complete!")
        st.write(f"Total Calories burned: {st.session_state.total_calories:.2f} cal")
        if st.button("Back to Menu"):
            go_to('menu')
    else:
        workout = st.session_state.schedule_queue[0]
        name, target_str = workout.split("–")
        name = name.strip()
        target = extract_target(target_str.strip())
        st.session_state.workout_name = name
        st.session_state.target_reps_or_time = target
        go_to('workout')

elif st.session_state.page == 'workout':
    st.header(f"Workout: {st.session_state.workout_name}")

    w = st.session_state.weight
    t = st.session_state.target_reps_or_time
    v = st.session_state.video_path

    calories = 0
    name_lower = st.session_state.workout_name.lower()

    try:
        if name_lower == "lunges":
            calories = run_lunges(w, t, v)
        elif name_lower == "pushups":
            calories = run_pushups(w, t, v)
        elif name_lower == "squats":
            calories = run_squats(w, t, v)
        elif "side-lying" in name_lower:
            calories = run_sidelying_leg_raises(w, t, v)
        elif name_lower == "plank":
            calories = run_plank(w, t, v)
        elif name_lower == "crunches":
            calories = run_crunches(w, t, v)
        elif "pose" in name_lower:
            calories = run_yoga_pose(w, t, v)
    except Exception as e:
        st.error(f"Error running workout: {e}")

    if isinstance(calories, dict):
        calories_value = calories.get("calories", 0)
    elif isinstance(calories, (int, float)):
        calories_value = calories
    else:
        calories_value = 0

    st.session_state.calories_per_workout.append(calories_value)
    st.session_state.total_calories = sum(st.session_state.calories_per_workout)

    st.subheader(f"Calories burned: {calories_value:.2f} cal")

    if "stop_detection" not in st.session_state:
      st.session_state.stop_detection = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Next Workout"):
        st.session_state.stop_detection = True
        st.session_state.selected_workout = None
        go_to("workout_selection")
        st.rerun()

with col2:
    if st.button("End Session"):
        st.session_state.stop_detection = True
        st.session_state.selected_workout = None
        st.session_state.schedule_queue = []
        go_to("menu")
        st.rerun()
