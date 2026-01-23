import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
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

FPS = 25

st.title("Analiza Meczu Piłkarskiego")
df = load_data()

possession = calculate_possession(df)
st.header("Posiadanie Piłki")

frames_team_a = possession.get(0,0)
frames_team_b = possession.get(1,0)

seconds_team_a = frames_team_a/FPS
seconds_team_b = frames_team_b/FPS

col1, col2 = st.columns(2)
col1.metric("Czas Team A", f"{seconds_team_a:.2f}")
col2.metric("Czas Team B", f"{seconds_team_b:.2f}")

team_names = {0: 'Team A', 1: 'Team B', -1: "Piłka"}
colors = {'Team A': 'red', 'Team B': 'blue'}

fig = px.pie(names=possession.index.map(team_names), values=possession.values, title="Possession Control", color=possession.index.map(team_names), color_discrete_map=colors)

st.plotly_chart(fig)


st.subheader("Analiza Średnich Pozycji  i Trajektoria Ruchu")

track_counts = df['track_id'].value_counts()

min_frame = 50
valid_tracks = track_counts[track_counts >= min_frame].index

df_clean = df[df['track_id'].isin(valid_tracks)].copy()

average_locs = df_clean.groupby('track_id').agg({
    'map_x': 'mean',
    'map_y': 'mean',
    'team_id': 'first',
    'frame_num': 'count'
}).reset_index()
average_locs = average_locs.sort_values('frame_num', ascending=False)

average_locs['team_name'] = average_locs['team_id'].map(team_names)

average_locs = average_locs.dropna(subset=['team_name'])
average_locs = average_locs.dropna(subset=['map_x', 'map_y'])

team_a_data = average_locs[average_locs['team_name'] == 'Team A'].head(11)
team_b_data = average_locs[average_locs['team_name'] == 'Team B'].head(11).copy()
team_b_data['map_x_rev'] = 105 - team_b_data['map_x']
team_b_data['map_y_rev'] = 68 - team_b_data['map_y']

pitch = Pitch(pitch_type='custom', pitch_length=105, pitch_width=68, line_color='white', pitch_color='#53803F')
fig, ax = pitch.draw(figsize=(10,6))

pitch.scatter(team_a_data['map_x'], team_a_data['map_y'], c='red', s=120, ax=ax)

pitch.scatter(team_b_data['map_x_rev'], team_b_data['map_y_rev'], c='blue', s=120, ax=ax)


ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=12)



tracks_a = df[df['track_id'].isin(team_a_data['track_id'])].copy()
tracks_b = df[df['track_id'].isin(team_b_data['track_id'])].copy()

tracks_b['map_x_rev'] = 105 - tracks_b['map_x']
tracks_b['map_y_rev'] = 68 - tracks_b['map_y']

for index, row in team_a_data.iterrows():
    player_path = tracks_a[tracks_a['track_id'] == row['track_id']]
    pitch.plot(player_path['map_x'], player_path['map_y'], color='#E63946', alpha=0.5,ax=ax)

for index, row in team_b_data.iterrows():
    player_path = tracks_b[tracks_b['track_id'] == row['track_id']]
    pitch.plot(player_path['map_x_rev'], player_path['map_y_rev'], color='#1D3557', alpha=0.5,ax=ax)

st.pyplot(fig)

tracks_a['dx'] = tracks_a.groupby('track_id')['map_x'].diff()
tracks_a['dy'] = tracks_a.groupby('track_id')['map_y'].diff()
tracks_a['distance'] = (tracks_a['dx'] **2 + tracks_a['dy']**2) ** 0.5
tracks_a['speed'] = (tracks_a['distance'] * FPS).rolling(window=12).mean()
tracks_a = tracks_a[(tracks_a['speed'] < 12) & (tracks_a['object_type'] != 'ball')]

tracks_b['dx'] = tracks_b.groupby('track_id')['map_x'].diff()
tracks_b['dy'] = tracks_b.groupby('track_id')['map_y'].diff()
tracks_b['distance'] = (tracks_b['dx'] ** 2 + tracks_b['dy'] ** 2) ** 0.5
tracks_b['speed'] = (tracks_b['distance']* FPS).rolling(window=12).mean()
tracks_b = tracks_b[(tracks_b['speed'] < 12) & (tracks_b['object_type'] != 'ball')]

max_speed_row_a = tracks_a[tracks_a['speed'] == tracks_a['speed'].max()]
max_speed_row_b = tracks_b[tracks_b['speed'] == tracks_b['speed'].max()]

fastest_player_id = max_speed_row_a['track_id'].values[0]
fastest_player_path = tracks_a[tracks_a['track_id'] == fastest_player_id]

st.subheader("Pozycja najszybszego zawodnika w meczu i trajektoria ruchu")

fig, ax = pitch.draw(figsize=(10,6))
pitch.plot(fastest_player_path['map_x'], fastest_player_path['map_y'], color='#E63946', alpha=0.5,ax=ax)
pitch.scatter(max_speed_row_a['map_x'], max_speed_row_a['map_y'], c='red', s=120, ax=ax)
ax.text(max_speed_row_a['map_x'].values[0], max_speed_row_a['map_y'].values[0], f"speed value {max_speed_row_a['speed'].max():.2f} m/s", fontsize=12)

st.pyplot(fig)

st.subheader("Analiza Taktyczna i Fizyczna")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Heatmapa Pozycji")
    fig, ax = pitch.draw(figsize=(10, 6))

    sns.kdeplot(
        x=tracks_a['map_x'],
        y=tracks_a['map_y'],
        fill=True,
        ax=ax,
        color='red',
        alpha=0.5,
    )
    sns.kdeplot(
        x=tracks_b['map_x'],
        y=tracks_b['map_y'],
        fill=True,
        ax=ax,
        color='blue',
        alpha=0.5
    )
    st.pyplot(fig)
with col2:
    st.subheader("Przebyty Dystans (top 5)")
    dist_a = tracks_a.groupby('track_id')['distance'].sum().sort_values(ascending=False).head(5)
    dist_b = tracks_b.groupby('track_id')['distance'].sum().sort_values(ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x=dist_a.index, y=dist_a.values, ax=ax, color='red')
    sns.barplot(x=dist_b.index, y=dist_b.values, ax=ax, color='blue')

    st.pyplot(fig)

st.divider()
st.subheader("Wnioski Analityczne")
st.success("""
1. **Znaczna dominacja Teamu B:** Widać to zarówno w posiadaniu piłki, jak i kontroli terytorialnej.
2. **Obrona Częstochowy:** Gra toczy się niemal wyłącznie pod bramką i w polu karnym Teamu A.
3. **Dysproporcja fizyczna:** Team A przebiega znacznie mniej kilometrów, co wynika z głębokiej defensywy i braku wyjść do kontrataku.
""")

