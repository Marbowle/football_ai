import streamlit as st
import pandas as pd
import plotly.express as px
from mplsoccer import Pitch

@st.cache_data()
def load_data():
    game_df = pd.read_csv("output_game_data.csv")
    return game_df

def calculate_possession(df):
    ball_df = df[df['object_type'] == 'ball']
    player_df = df[df['object_type'] == 'players']

    player_ball_df = pd.merge(player_df, ball_df, on='frame_num', suffixes=['_players', '_ball'])
    x1 = player_ball_df['map_x_players']
    y1 = player_ball_df['map_y_players']
    x2 = player_ball_df['map_x_ball']
    y2 = player_ball_df['map_y_ball']

    distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    player_ball_df['distance'] = distance

    active_possession = player_ball_df[player_ball_df['distance'] <= 4]

    active_possession = active_possession.sort_values('distance').drop_duplicates(subset='frame_num', keep='first')

    ball_df = pd.merge(ball_df, active_possession ,how ='left' ,on='frame_num', suffixes=['_ball', '_players'])

    ball_df = ball_df.ffill()
    possession_counts = ball_df['team_id_players'].value_counts()

    return possession_counts


st.title("Analiza Meczu Piłkarskiego")
df = load_data()

possession = calculate_possession(df)
st.header("Posiadanie Piłki")

frames_team_a = possession.get(0,0)
frames_team_b = possession.get(1,0)

seconds_team_a = frames_team_a/25
seconds_team_b = frames_team_b/25

col1, col2 = st.columns(2)
col1.metric("Czas Team A", f"{seconds_team_a:.2f}")
col2.metric("Czas Team B", f"{seconds_team_b:.2f}")

team_names = {0: 'Team A', 1: 'Team B', -1: "Piłka"}

fig = px.pie(names=possession.index.map(team_names), values=possession.values, title="Possession Control")

st.plotly_chart(fig)


st.subheader("Analiza Średnich Pozycji (Average Positions)")

track_counts = df['track_id'].value_counts()

min_frame = 50
valid_tracks = track_counts[track_counts >= min_frame].index

df_clean = df[df['track_id'].isin(valid_tracks)].copy()

average_locs = df_clean.groupby('track_id').agg({
    'map_x': 'mean',
    'map_y': 'mean',
    'team_id': 'first'
})

average_locs['team_name'] = average_locs['team_id'].map(team_names)

average_locs = average_locs.dropna(subset=['team_name'])
average_locs = average_locs.dropna(subset=['map_x', 'map_y'])

team_a_data = average_locs[average_locs['team_name'] == 'Team A']
team_b_data = average_locs[average_locs['team_name'] == 'Team B']

pitch = Pitch(pitch_type='custom', pitch_length=105, pitch_width=68, line_color='white', pitch_color='#53803F')
fig, ax = pitch.draw(figsize=(10,6))

pitch.scatter(team_a_data['map_x'], team_a_data['map_y'], c='red', s=120, ax=ax)

pitch.scatter(team_b_data['map_x'], team_b_data['map_y'], c='blue', s=120, ax=ax)

st.pyplot(fig)

