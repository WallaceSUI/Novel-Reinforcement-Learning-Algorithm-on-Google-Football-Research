

from kaggle_environments.envs.football.helpers import *
from math import sqrt
import numpy as np

def get_postion_single(obs):
    shp = obs.shape
    for i in range(shp[0]):
        for j in range(shp[1]):
            if (obs[i,j] > 0):
                return [j/float(shp[1]),i/float(shp[0])]

directions = [
[2, 3, 4],
[1, 0, 5],
[8, 7, 6]]

dirsign = lambda x: 1 if abs(x) < 0.01 else (0 if x < 0 else 2)

enemyGoal = [1, 0]
perfectRange = [[0.61, 1], [-0.2, 0.2]]

def inside(pos, area):
    return area[0][0] <= pos[0] <= area[0][1] and area[1][0] <= pos[1] <= area[1][1]

def get_distance(pos1,pos2):
    return ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5

class SMMFrameProcessWrapper(gym.Wrapper):
    """
    Wrapper for processing frames from SMM observation wrapper from football env.

    Input is (72, 96, 4), where last dim is (team 1 pos, team 2 pos, ball pos, 
    active player pos). Range 0 -> 255.
    Output is (72, 96, 4) as difference to last frame for all. Range -1 -> 1
    """

    def __init__(self, env: gym.Env = None,
                 obs_shape: Tuple[int, int] = (72, 96, 4)) -> None:
        """
        :param env: Gym env, or None. Allowing None here is unusual,
                    but we'll reuse the buffer functunality later in
                    the submission, when we won't be using the gym API.
        :param obs_shape: Expected shape of single observation.
        """
        if env is not None:
            super().__init__(env)
        self._buffer_length = 2
        self._obs_shape = obs_shape
        self._prepare_obs_buffer()

    @staticmethod
    def _normalise_frame(frame: np.ndarray):
        return frame / 255.0

    def _prepare_obs_buffer(self) -> None:
        """Create buffer and preallocate with empty arrays of expected shape."""

        self._obs_buffer = collections.deque(maxlen=self._buffer_length)

        for _ in range(self._buffer_length):
            self._obs_buffer.append(np.zeros(shape=self._obs_shape))

    def build_buffered_obs(self) -> np.ndarray:
        """
        Iterate over the last dimenion, and take the difference between this obs 
        and the last obs for each.
        """
        agg_buff = np.empty(self._obs_shape)
        for f in range(self._obs_shape[-1]):
            agg_buff[..., f] = self._obs_buffer[1][..., f] - self._obs_buffer[0][..., f]

        return agg_buff


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:

        """Step env, add new obs to buffer, return buffer."""
        obs, reward, done, info = self.env.step(action)
        # print(obs)
        obs = self._normalise_frame(obs)
        controlled_player_pos = get_postion_single(obs[:,:,3]) # active player 
        ball_pos = get_postion_single(obs[:,:,2]) # ball
        # Make sure player is running.
        if  0 < controlled_player_pos[0] < 0.6:
            if (action == 13): # Sprint
                reward += 0.01
        elif 0.6 < controlled_player_pos[0]:
            if (action == 8): # Release Sprint
                reward += 0.01
        
        if get_distance(controlled_player_pos,ball_pos) < 0.01:
            goalkeeper = 0
            #if in the zone near goal shoot
            if inside(controlled_player_pos, perfectRange) and controlled_player_pos[0] < ball_pos[0]:
                if (action == 12): # shot
                    reward += 0.01
            #if the goalie is coming out on player near goal shoot
            elif controlled_player_pos[0] > 0.4 and abs(controlled_player_pos[1]) < 0.2:
                if (action == 12): # shot
                    reward += 0.01
            # if close to goal and too wide for shot pass the ball
            elif controlled_player_pos[0] >.75 and controlled_player_pos[1] >.20 or controlled_player_pos[0] >.75 and controlled_player_pos[1] <-.20 :
                if (action == 11): # short pass
                    reward += 0.01
            # if near our goal and moving away long pass to get out of our zone
            elif controlled_player_pos[0]<-.3:
                if (action == 9): # long pass
                    reward += 0.01
        else: 
            #vector where ball is going
            ball_targetx=ball_pos[0]
            ball_targety=ball_pos[1]
        
            #euclidian distance to the ball so we head off movement until very close
            e_dist=get_distance(controlled_player_pos,ball_pos)
            #if not close to ball move to where it is going
            if e_dist >.005:
                # Run where ball will be
                xdir = dirsign(ball_targetx - controlled_player_pos[0])
                ydir = dirsign(ball_targety - controlled_player_pos[1])
                if (action== directions[ydir][xdir]):
                    reward += 0.01
            #if close to ball go to ball
            else:
                # Run towards the ball.
                xdir = dirsign(ball_pos[0] - controlled_player_pos[0])
                ydir = dirsign(ball_pos[1] - controlled_player_pos[1])
                if (action== directions[ydir][xdir]):
                    reward += 0.01


        self._obs_buffer.append(obs)
        if action == 0:
            reward -= 10 #idle
        return self.build_buffered_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        """Add initial obs to end of pre-allocated buffer.

        :return: Buffered observation
        """
        self._prepare_obs_buffer()
        obs = self.env.reset()
        self._obs_buffer.append(obs)

        return self.build_buffered_obs()