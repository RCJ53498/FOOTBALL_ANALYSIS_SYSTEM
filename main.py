from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import os
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from passing_network_analyzer import PassingNetworkAnalyzer  # Import the new module
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg to avoid display issues
import matplotlib.pyplot as plt

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')
    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
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
    
    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 'unknown')
    team_ball_control = np.array(team_ball_control)
    
    # Create output directories if they don't exist
    os.makedirs('output_images', exist_ok=True)
    os.makedirs('output_videos', exist_ok=True)
    
    # Initialize Passing Network Analyzer
    passing_analyzer = PassingNetworkAnalyzer()
    
    # Detect passes
    passes = passing_analyzer.detect_passes(tracks)
    print(f"Detected {len(passes)} passes in the video")
    
    # Set team colors for visualization
    team1_color = (0, 255, 255)  # Yellow in BGR format (OpenCV uses BGR)
    team2_color = (255, 255, 255) 
    passing_analyzer.team_colors = {'team1': team1_color,'team2': team2_color}
    
    # Create passing networks for both teams
    team1_graph, team1_positions = passing_analyzer.create_passing_network(tracks, 'team1')
    team2_graph, team2_positions = passing_analyzer.create_passing_network(tracks, 'team2')
    
    # Generate passing statistics
    team1_stats = passing_analyzer.generate_team_passing_statistics(tracks, 'team1')
    team2_stats = passing_analyzer.generate_team_passing_statistics(tracks, 'team2')
    
    print(f"Team 1 Stats: {team1_stats['total_passes']} passes")
    print(f"Team 2 Stats: {team2_stats['total_passes']} passes")
    
    # Draw passing networks for both teams
    plt.ioff()  # Turn off interactive mode
    
    team1_fig = passing_analyzer.draw_passing_network(team1_graph, team1_positions, team1_color)
    team2_fig = passing_analyzer.draw_passing_network(team2_graph, team2_positions, team2_color)
    
    # Save the network visualizations as separate images
    team1_fig.savefig('output_images/team1_passing_network.png', bbox_inches='tight', facecolor=team1_fig.get_facecolor())
    team2_fig.savefig('output_images/team2_passing_network.png', bbox_inches='tight', facecolor=team2_fig.get_facecolor())
    
    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    ## Draw Speed and Distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    ## Draw passing information
    for frame_num in range(len(output_video_frames)):
        # Draw recent passes on the frame
        output_video_frames[frame_num] = passing_analyzer.draw_live_passes(
            output_video_frames[frame_num], 
            tracks, 
            frame_num
        )
        
        # Add network visualization overlay for alternating teams
        if frame_num % 200 < 100:  # Switch visualization every 100 frames
            fig_to_use = team1_fig
            team_name = "Team 1"
            stats = team1_stats
        else:
            fig_to_use = team2_fig
            team_name = "Team 2"
            stats = team2_stats
            
        output_video_frames[frame_num] = passing_analyzer.render_network_to_frame(
            output_video_frames[frame_num],
            fig_to_use
        )
        
        # Print passing statistics on the frame
        cv2.putText(
    output_video_frames[frame_num],
    f"{team_name} Passes: {stats['total_passes']}", 
    (output_video_frames[frame_num].shape[1] - 250, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
)
    
    # Save video
    save_video(output_video_frames, 'output_videos/output_video_with_passing.avi')
    
    # Clean up matplotlib figures
    plt.close(team1_fig)
    plt.close(team2_fig)
    plt.close('all')

if __name__ == '__main__':
    main()