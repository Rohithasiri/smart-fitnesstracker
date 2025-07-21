import streamlit as st
from lunges import run_lunges
from pushups import run_pushups
from squats import run_squats
from sidelyinglegraises import run_sidelying_leg_raises as run_side_lying_leg_raises
from plank import run_plank
from yoga_pose_classifier import run_yoga_pose

if "results" not in st.session_state:
    st.session_state["results"] = []

# Initialize session state if not already set
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

def go_to(page):
    st.session_state.page = page
    st.rerun()

# -------------------- PAGE: Welcome --------------------
if st.session_state.page == 'welcome':
    st.title("🏋️ Welcome to GetUpGo! the Smart Fitness Tracker!")
    st.write("Track exercises and posture using vision-based pose detection!")
    
    camera_indices = [0,1,2,3]
    selected_camera = st.selectbox("Select a webcam source:", camera_indices)

    st.session_state.video_path = selected_camera
    
    if st.button("Start Workout"):        
        go_to('select')
        

# -------------------- PAGE: Select --------------------
elif st.session_state.page == 'select':
    st.title("Select Your Workout")

    option = st.selectbox("Choose an exercise", [
        "Lunges", "Pushups", "Squats", "Side-Lying Leg Raises", "Plank", "Yoga"])
    weight = st.number_input("Enter your weight (kg)", min_value=30, max_value=150, value=60)

    video_path = st.session_state.video_path
    st.session_state.weight = weight
    st.session_state.exercise = option
    st.session_state.video = video_path

    if option == "Yoga":
        yoga_pose = st.selectbox("Select Yoga Pose", ["T Pose", "Warrior II Pose", "Chair Pose"])
        hold_time = st.number_input("Target hold time (in seconds)", min_value=5, max_value=120, value=30)
        st.session_state.yoga_pose = yoga_pose
        st.session_state.hold_time = hold_time
    else:
        target_reps = st.number_input("Target repetitions", min_value=1, max_value=100, value=5)
        st.session_state.target_reps = target_reps

    if st.button("Start Exercise"):
        go_to('workout')

# -------------------- PAGE: Workout --------------------
elif st.session_state.page == 'workout':
    st.title(f"🏃 Starting {st.session_state.exercise}!")
    st.warning("Workout is running in a separate window. Please wait...")

    result = None
    if st.session_state.exercise == "Lunges":
        result = run_lunges(
            user_weight=st.session_state.weight,
            video_path=st.session_state.video,
            target_reps=st.session_state.target_reps
        )
        
    elif st.session_state.exercise == "Pushups":
        result = run_pushups(
            user_weight=st.session_state.weight,
            video_path=st.session_state.video,
            target_reps=st.session_state.target_reps
        )
    elif st.session_state.exercise == "Squats":
        result = run_squats(
            user_weight=st.session_state.weight,
            video_path=st.session_state.video,
            target_reps=st.session_state.target_reps
        )
    elif st.session_state.exercise == "Side-Lying Leg Raises":
        result = run_side_lying_leg_raises(
            user_weight=st.session_state.weight,
            video_path=st.session_state.video,
            target_reps=st.session_state.target_reps
        )
    elif st.session_state.exercise == "Plank":
        result = run_plank(
            user_weight=st.session_state.weight,
            video_path=st.session_state.video,
            target_time=st.session_state.target_reps  # reused for plank time
        )
    elif st.session_state.exercise == "Yoga":
        result = run_yoga_pose(
            user_weight=st.session_state.weight,
            video_path=st.session_state.video,
            target_time=st.session_state.hold_time,
            pose_name=st.session_state.yoga_pose
        )
    
    st.session_state["results"].append(result)
    st.session_state.result = result
    go_to('result')
    

# # -------------------- PAGE: Result --------------------
# elif st.session_state.page == 'result':
#     print("Result")
#     result = st.session_state.result
#     st.title("✅ Workout Completed")

#     if result['status'] == "Success":
#         st.success(f"Great job! You completed your {st.session_state.exercise} workout.")
#         if 'reps' in result:
#             st.write(f"Repetitions: {result['reps']}")
#         if 'calories' in result:
#             st.write(f"Calories Burned: {result['calories']} cal")
#         if 'time' in result:
#             st.write(f"Hold Time: {result['time']} seconds")
#     else:
#         print(result['status'])
#         st.error("Workout failed. Please try again.")

#     if st.button("Do another workout"):
#         go_to('select')
#     elif st.button("End session"):
#         go_to('welcome')
# -------------------- PAGE: Result --------------------
elif st.session_state.page == 'result':
    print("Result")
    result = st.session_state.result
    st.title("✅ Workout Completed")

    # Initialize result history if not present
    if "results" not in st.session_state:
        st.session_state["results"] = []

    # Append current result to history only if it’s not already there
    if result not in st.session_state["results"]:
        st.session_state["results"].append(result)

    # Show current workout result
    if result['status'] == "Success":
        st.success(f"Great job! You completed your {st.session_state.exercise} workout.")
        if 'reps' in result:
            st.write(f"Repetitions: {result['reps']}")
        if 'calories' in result:
            st.write(f"Calories Burned: {result['calories']} cal")
        if 'time' in result:
            st.write(f"Hold Time: {result['time']} seconds")
    else:
        print(result['status'])
        st.error("Workout failed. Please try again.")

    st.divider()
    st.subheader("📜 Workout History")

    total_calories = 0
    for i, res in enumerate(st.session_state["results"], 1):
        st.markdown(f"### 🏋️‍♀️ Workout {i}")
        if not res.get('exercise') == 'Yoga':
            st.write(f"**Exercise:** {res.get('exercise', 'Unknown')}")
        else :
            st.write(f"**Yoga Asana:** {res.get('pose','Unknown')}")
        if 'reps' in res:
            st.write(f"**Repetitions:** {res['reps']}")
        if 'time' in res:
            st.write(f"**Hold Time:** {res['time']} seconds")
        if 'calories' in res:
            st.write(f"**Calories Burned:** {res['calories']} cal")
            total_calories += res['calories']
        st.divider()

    st.success(f"🔥 **Total Calories Burned:** {round(total_calories, 2)} cal")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 Do another workout"):
            go_to('select')
    with col2:
        if st.button("❌ End session"):
            st.session_state["results"] = []  # Clear history
            go_to('welcome')
