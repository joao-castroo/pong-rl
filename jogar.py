"""Sessão humana para o requisito de familiarização. W/S ou setas; espaço; Esc."""
from pathlib import Path
import json, time
import pygame
import gymnasium as gym
import ale_py

def main():
    root=Path(__file__).resolve().parent
    (root/'artifacts').mkdir(exist_ok=True)
    gym.register_envs(ale_py)
    pygame.init()
    screen=pygame.display.set_mode((640,840))
    pygame.display.set_caption('Pong — W/S ou setas, espaço, Esc')
    clock=pygame.time.Clock()
    env=gym.make('ALE/Pong-v5',frameskip=4,render_mode='rgb_array',repeat_action_probability=0.25)
    obs,_=env.reset(seed=42)
    start=time.perf_counter(); ret=0; games=[]; steps=0
    try:
        running=True
        while running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
                if event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE: running=False
            keys=pygame.key.get_pressed()
            up=keys[pygame.K_w] or keys[pygame.K_UP]
            down=keys[pygame.K_s] or keys[pygame.K_DOWN]
            fire=keys[pygame.K_SPACE]
            action=(4 if fire else 2) if up else ((5 if fire else 3) if down else (1 if fire else 0))
            obs,reward,terminated,truncated,_=env.step(action)
            steps+=1; ret+=float(reward)
            surface=pygame.surfarray.make_surface(obs.transpose(1,0,2))
            screen.blit(pygame.transform.scale(surface,(640,840)),(0,0)); pygame.display.flip()
            if terminated or truncated:
                games.append(ret); ret=0; obs,_=env.reset()
            clock.tick(15)
    finally:
        env.close(); pygame.quit()
        result=dict(seconds=time.perf_counter()-start, decisions=steps,
                    completed_game_returns=games, unfinished_game_return=ret,
                    note='Relato pessoal ainda deve ser preenchido no notebook.')
        (root/'artifacts/sessao_humana.json').write_text(json.dumps(result,indent=2))
        print(result)

if __name__=='__main__': main()
