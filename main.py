from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assignment import PlayerBallAssigner


def main():
    video_frames=read_video('input_videos/08fd33_4.mp4')

    #init 
    tracker=Tracker('model/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/tracks_stubs.pkl')
    

    #Interpolate
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])




    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]



    # Parameters
    MIN_POSSESSION_FRAMES = 12  # Adjust for 1 sec (30 frames if video is 30 FPS)

    # Initialize ball assigner
    player_assigner = PlayerBallAssigner()

    # Tracking variables
    team_ball_control = []
    last_team = None
    possession_counter = 0
    player_possession_frames = {}  # player_id: consecutive possession count

    # Main loop
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            player_id = assigned_player
            player_team = player_track[player_id]['team']

            # Increment possession count for this player
            if player_id not in player_possession_frames:
                player_possession_frames[player_id] = 1
            else:
                player_possession_frames[player_id] += 1

            # Reset others' counters and has_ball
            for other_id in player_track:
                if other_id != player_id:
                    player_possession_frames[other_id] = 0
                    player_track[other_id]['has_ball'] = False

            # Only mark possession if sustained
            if player_possession_frames[player_id] >= MIN_POSSESSION_FRAMES:
                player_track[player_id]['has_ball'] = True

                # Team control logic
                if player_team == last_team:
                    possession_counter += 1
                else:
                    possession_counter = 1

                if possession_counter >= MIN_POSSESSION_FRAMES:
                    team_ball_control.append(player_team)
                else:
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else player_team)

                last_team = player_team
            else:
                player_track[player_id]['has_ball'] = False
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)

        else:
            # No player assigned
            for pid in player_track:
                player_track[pid]['has_ball'] = False
            possession_counter = 0
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)

    # Final result: team control array
    team_ball_control = np.array(team_ball_control)




            

    #draw
    output_video_frames = tracker.draw_anotations(video_frames, tracks, team_ball_control)

  


    save_video(output_video_frames,'output_videos/output_video.avi')

if __name__=="__main__":
    main()  