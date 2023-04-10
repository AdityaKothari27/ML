import pymunk
import math
import numpy as np
import pickle as pcl
from SynapEvo.NNevo import Population
import matplotlib.pyplot as plt
from tqdm import tqdm
class Game:
    def __init__(self) -> None:
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 678
        self.BOTTOM_PANEL = 50
        self.space = pymunk.Space()
        self.static_body = self.space.static_body
        self.FPS = 120
        self.lives = 3
        self.dia = 36
        self.pocket_dia = 70
        self.force = 0
        self.max_force = 10000
        self.force_direction = 1
        self.game_running = True
        self.cue_ball_potted = False
        self.taking_shot = True
        self.powering_up = False
        self.potted_balls = []
        self.balls = []
        self.rows = 5
        self.gameover = False
        # potting balls
        for col in range(5):
            for row in range(self.rows):
                pos = (250 + (col * (self.dia + 1)), 267 + (row * (self.dia + 1)) + (col * self.dia / 2))
                new_ball = self.create_ball(self.dia / 2, pos)
                self.balls.append(new_ball)
            self.rows -= 1
        # cue ball
        pos = (888, self.SCREEN_HEIGHT / 2)
        cue_ball = self.create_ball(self.dia / 2, pos)
        self.balls.append(cue_ball)
        # create six pockets on table
        self.pockets = [
            [55, 63],
            [592, 48],
            [1134, 64],
            [55, 616],
            [592, 629],
            [1134, 616]
        ]

        # create pool table cushions
        cushions = [
            [(88, 56), (109, 77), (555, 77), (564, 56)],
            [(621, 56), (630, 77), (1081, 77), (1102, 56)],
            [(89, 621), (110, 600), (556, 600), (564, 621)],
            [(622, 621), (630, 600), (1081, 600), (1102, 621)],
            [(56, 96), (77, 117), (77, 560), (56, 581)],
            [(1143, 96), (1122, 117), (1122, 560), (1143, 581)]
        ]

        for c in cushions:
            self.create_cushion(c)
# function for creating balls
    def create_ball(self,radius, pos):
        body = pymunk.Body()
        body.position = pos
        shape = pymunk.Circle(body, radius)
        shape.mass = 5
        shape.elasticity = 0.8
        # use pivot joint to add friction
        pivot = pymunk.PivotJoint(self.static_body, body, (0, 0), (0, 0))
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = 1000  # emulate linear friction
        self.space.add(body, shape, pivot)
        return shape
    # function for creating cushions
    def create_cushion(self,poly_dims):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = ((0, 0))
        shape = pymunk.Poly(body, poly_dims)
        shape.elasticity = 0.8
        self.space.add(body, shape)
    def scoreState(self,cue_angle,force):
        x_impulse = math.cos(cue_angle)
        y_impulse = math.sin(cue_angle)
        score = 0.0
        if(force>1):
            force = self.max_force
            score-=20
        elif(force < 0):
            force = 0
            score-=50
        else:
            force *= self.max_force
        self.balls[-1].body.apply_impulse_at_local_point((force * -x_impulse, force * y_impulse), (0, 0))
        self.space.step(1/self.FPS)
        check = True
        score = 0.0
        max_iter = 500
        while check and max_iter > 0:
            self.space.step(1/self.FPS)
            s = score
            if len(self.balls) == 1:
                self.gameover = True
                check = True
                break
            elif self.lives < 0:
                self.gameover = True
                check = True
                break
            else:
                for i, ball in enumerate(self.balls):
                    for pocket in self.pockets:
                        ball_x_dist = abs(ball.body.position[0] - pocket[0])
                        ball_y_dist = abs(ball.body.position[1] - pocket[1])
                        ball_dist = math.sqrt((ball_x_dist ** 2) + (ball_y_dist ** 2))
                        if ball_dist <= self.pocket_dia / 2:
                            # check if the potted ball was the cue ball
                            if i == len(self.balls) - 1:
                                self.lives -= 1
                                self.cue_ball_potted = True
                                ball.body.position = (-100, -100)
                                ball.body.velocity = (0.0, 0.0)
                                score-=100
                            else:
                                self.space.remove(ball.body)
                                self.balls.remove(ball)
                                score+=500
                if score == s:
                    score -= 50
                stop = True
                for ball in self.balls:
                    if (int(ball.body.velocity[0]) != 0 or int(ball.body.velocity[1]) != 0):
                        stop = False
                if stop:
                    
                    
                    check = False
        if self.cue_ball_potted == True:
                    
                        # reposition cue ball
                    
                        self.balls[-1].body.position = (888, self.SCREEN_HEIGHT / 2)
                        self.cue_ball_potted = False    
        max_iter -= 1
        return score

def inputFormatter(GameObj):
    pockets = GameObj.pockets
    balls = []
    for ball in GameObj.balls:
        balls.append(ball.body.position)
    input = pockets+balls
    input = np.array(input).flatten()
    return input


Pop = Population(input_size = 44,output_size = 2,layers_sizes = 22 , nlayers = 11,np_nr = 0.60,population_size = 100,parent_percentage = 0.4,mutation_rate = 0.7)
ourpop = Pop.get_populations()
aths = 0
athss = None
hsgen = []

for i in range(50):
    # running for about 50 generations
    hs = 0
    print(f"Generation {i}:")
    for j in tqdm(range(len(ourpop))):
        game = Game()
        score = 0
        try:
            for k in range(100):
                arr = ourpop[j].forward(inputFormatter(game))
                score += game.scoreState(arr[0],arr[1])
                if game.gameover:
                    break
            ourpop[j].score = score
            if score >= hs:
                hs = score
            if score >= aths:
                aths = score
                athss = ourpop[j]
        except:
            score = -9999999
            continue
    print(f"Best Fitness:{hs}")
    hsgen.append(hs)
    ourpop = Pop.evolve(ourpop)
print(aths)
game = Game()
import pygame
pygame.init()


# game window
screen = pygame.display.set_mode((game.SCREEN_WIDTH, game.SCREEN_HEIGHT + game.BOTTOM_PANEL))
pygame.display.set_caption("Pool")
# colours
BG = (50, 50, 50)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
# fonts
font = pygame.font.SysFont("Lato", 30)
large_font = pygame.font.SysFont("Lato", 60)
# load images
cue_image = pygame.image.load("images/cue.png").convert_alpha()
table_image = pygame.image.load("images/table.png").convert_alpha()
ball_images = []
for i in range(1, 17):
    ball_image = pygame.image.load(f"images/ball_{i}.png").convert_alpha()
    ball_images.append(ball_image)

# function for outputting text onto the screen
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))
# create pool cue
class Cue():
    def __init__(self, pos):
        self.original_image = cue_image
        self.angle = 0
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = pos

    def update(self, angle):
        self.angle = angle

    def draw(self, surface):
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        surface.blit(self.image,
                     (self.rect.centerx - self.image.get_width() / 2,
                      self.rect.centery - self.image.get_height() / 2)
                     )
cue = Cue(game.balls[-1].body.position)
# create power bars to show how hard the cue ball will be hit
power_bar = pygame.Surface((10, 20))
power_bar.fill(RED)
gameon = True
inmotion = False
while gameon:
    # fill background
    screen.fill(BG)
    # draw pool table
    screen.blit(table_image, (0, 0))
    
    for i, ball in enumerate(game.balls):
        screen.blit(ball_images[i], (ball.body.position[0] - ball.radius, ball.body.position[1] - ball.radius))
    if not inmotion:
        arr = athss.forward(inputFormatter(game))
        x_impulse = math.cos(arr[0])
        y_impulse = math.sin(arr[1])
        
        if(force>game.max_force):
            force =game.max_force
            
        if(force < 0):
            force = 0
        else:    
            force *= game.max_force
        game.balls[-1].body.apply_impulse_at_local_point((force * -x_impulse, force * y_impulse), (0, 0))
    if inmotion:
        game.space.step(1/game.FPS)
        for i, ball in enumerate(game.balls):
            screen.blit(ball_images[i], (ball.body.position[0] - ball.radius, ball.body.position[1] - ball.radius))
        for i, ball in enumerate(game.balls):
            for pocket in game.pockets:
                ball_x_dist = abs(ball.body.position[0] - pocket[0])
                ball_y_dist = abs(ball.body.position[1] - pocket[1])
                ball_dist = math.sqrt((ball_x_dist ** 2) + (ball_y_dist ** 2))
                if ball_dist <= game.pocket_dia / 2:
                    # check if the potted ball was the cue ball
                    if i == len(game.balls) - 1:
                        game.lives -= 1
                        cue_ball_potted = True
                        ball.body.position = (-100, -100)
                        ball.body.velocity = (0.0, 0.0)
                    else:
                        game.space.remove(ball.body)
                        game.balls.remove(ball)
                        game.potted_balls.append(ball_images[i])
                        ball_images.pop(i)
        stop = True
        for ball in game.balls:
            if int(ball.body.velocity[0]) != 0 or int(ball.body.velocity[1]) != 0:
                stop = False
        if stop:
            game.balls[-1].body.position = (888, game.SCREEN_HEIGHT / 2)
            game.cue_ball_potted = False
            inmotion = False
         # draw bottom panel
    pygame.draw.rect(screen, BG, (0, game.SCREEN_HEIGHT, game.SCREEN_WIDTH, game.BOTTOM_PANEL))
    draw_text("LIVES: " + str(game.lives), font, WHITE, game.SCREEN_WIDTH - 200, game.SCREEN_HEIGHT + 10)

    # draw potted balls in bottom panel
    for i, ball in enumerate(game.potted_balls):
        screen.blit(ball, (10 + (i * 50), game.SCREEN_HEIGHT + 10))
    # check for game over
    if game.lives <= 0:
        draw_text("GAME OVER", large_font, WHITE, game.SCREEN_WIDTH / 2 - 160, game.SCREEN_HEIGHT / 2 - 100)
        gameon = False

    # check if all balls are potted
    if len(game.balls) == 1:
        draw_text("YOU WIN!", large_font, WHITE, game.SCREEN_WIDTH / 2 - 160, game.SCREEN_HEIGHT / 2 - 100)
        gameon = False
    pygame.display.update()
pygame.quit()