import pygame
import sys
import main as fl

dataset = [[0, 1, 0, 0],
           [1, 1, 1, 1],
           [0, 0, 1, 0],
           [1, 0, 1, 1]]

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = fl.initialize_network(n_inputs, 2, n_outputs)
fl.train_network(network, dataset, 0.5, 10000, n_outputs)

clock = pygame.time.Clock()
pygame.init()
screen = pygame.display.set_mode([800, 800], pygame.NOFRAME)

locked = pygame.image.load("LockSharpIcon.png")
unlocked = pygame.image.load("TickIcon.png")
locked = pygame.transform.scale(locked, (30, 30))
unlocked = pygame.transform.scale(unlocked, (30, 30))
display_img = locked

display_dataset = False
counter1 = 0


def main_menu():
    font = pygame.font.Font("CaviarDreams.ttf", 20)
    font_25 = pygame.font.Font("CaviarDreams.ttf", 25)
    font_20 = pygame.font.Font("CaviarDreams.ttf", 20)
    font_25_bold = pygame.font.Font("CaviarDreams_Bold.ttf", 25)
    user_text = ''
    text_rect = pygame.Rect(50, 120, 140, 32)
    color_active = pygame.Color('lightskyblue3')
    color_passive = pygame.Color('gray15')
    color = color_passive
    global display_dataset

    bg = pygame.image.load("Landscape.png")

    i = 0

    messages = ["Input first number", "Input second number", "Input third number"]

    explanation = "This neural network has been trained to respond positively to 1's in the first inputs"
    option = "(you can see the dataset we trained it with in the settings)"

    inputs = []
    active = False
    button = pygame.Rect(750, 25, 30, 30)
    settings_img = pygame.image.load("Settings.png")
    settings_img = pygame.transform.scale(settings_img, (30, 30))
    exit = pygame.image.load("Exit Right.png")
    exit = pygame.transform.scale(exit, (30, 30))
    exit_rect = pygame.Rect(705, 25, 30, 30)

    while True:

        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if text_rect.collidepoint(event.pos):
                    active = True
                elif button.collidepoint(event.pos):
                    print("Hey")
                    settings()
                elif exit_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

                else:
                    active = False
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    elif event.key == pygame.K_RETURN:
                        inputs.append(int(user_text))
                        user_text = ""
                        i += 1
                    else:
                        user_text += event.unicode
        screen.blit(bg, (0, 0))
        # screen.fill((0, 0, 0))

        if active:
            color = color_active
        else:
            color = color_passive

        title = font_25.render("BASIC NEURAL NETWORK", True, (255, 255, 255))
        output_text = font_25.render("OUTPUT:", True, (255, 255, 255))
        text_surface = font.render(user_text, True, (255, 255, 255))
        pygame.draw.rect(screen, color, text_rect, 2)
        screen.blit(text_surface, (text_rect.x + 5, text_rect.y + 5))
        screen.blit(title, (20, 20))
        screen.blit(output_text, (20, 220))
        text_rect.w = max((100, text_surface.get_width() + 10))

        # pygame.draw.rect(screen, (255, 255, 255), button)
        screen.blit(settings_img, (750, 25))

        if i <= 2:
            message = font_25.render(messages[i], True, (255, 255, 255))
            screen.blit(message, (text_rect.x, text_rect.y - 30))
        else:
            active = False
            prediction = fl.predict(network, inputs)
            output_result = font_25_bold.render(str(prediction), True, (255, 255, 255))
            screen.blit(output_result, (130, 220))
            message = font_25.render("All inputs received!", True, (255, 255, 255))
            screen.blit(message, (text_rect.x, text_rect.y - 30))

        if display_dataset:
            row1 = font_25.render("[0, 1, 0, 0]", True, (255, 255, 255))
            row2 = font_25.render("[1, 1, 1, 1]", True, (255, 255, 255))
            row3 = font_25.render("[0, 0, 1, 0]", True, (255, 255, 255))
            row4 = font_25.render("[1, 0, 1, 1]", True, (255, 255, 255))
            data_text = font_25.render("DATASET:", True, (255, 255, 255))
            screen.blit(row1, (650, 100))
            screen.blit(row2, (650, 140))
            screen.blit(row3, (650, 180))
            screen.blit(row4, (650, 220))
            screen.blit(data_text, (535, 160))

        screen.blit(exit, (705, 25))
        explanation_render = font_20.render(explanation, True, (255, 255, 255))
        option_render = font_20.render(option, True, (255, 255, 255))
        screen.blit(explanation_render, (22, 300))
        screen.blit(option_render, (22, 330))
        pygame.display.flip()
        clock.tick(60)
        print(display_dataset)


def settings():
    global display_dataset
    global counter1
    global display_img

    font = pygame.font.Font("CaviarDreams.ttf", 20)
    font_25 = pygame.font.Font("CaviarDreams.ttf", 25)
    font_25_bold = pygame.font.Font("CaviarDreams_Bold.ttf", 25)

    title = font_25_bold.render("SETTINGS", True, (255, 255, 255))
    render_weights = font_25.render("DISPLAY DATASET", True, (255, 255, 255))

    button = pygame.Rect(750, 25, 30, 30)
    settings_img = pygame.image.load("Settings.png")
    settings_img = pygame.transform.scale(settings_img, (30, 30))

    running = True
    menu = pygame.image.load("44773.png")
    menu = pygame.transform.scale(menu, (400, 600))

    weights_button = pygame.Rect(455, 300, 30, 30)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button.collidepoint(event.pos):
                    running = False
                if weights_button.collidepoint(event.pos):
                    counter1 += 1
                    if counter1 % 2 == 1:
                        weights = True
                        display_img = unlocked
                        display_dataset = weights
                    else:
                        weights = False
                        display_img = locked
                        display_dataset = weights
        screen.blit(settings_img, (750, 25))
        screen.blit(menu, (200, 150))
        screen.blit(render_weights, (250, 300))
        screen.blit(title, (350, 200))

        # pygame.draw.rect(screen, (255, 255, 255), weights_button)
        screen.blit(display_img, (455, 300))

        print(counter1)
        pygame.display.flip()
        clock.tick(60)


main_menu()
