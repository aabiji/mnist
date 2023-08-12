extern crate sdl2;
use sdl2::ttf::Font;
use sdl2::rect::Rect;
use sdl2::event::Event;
use sdl2::pixels::Color;
use sdl2::keyboard::Keycode;
use sdl2::render::{WindowCanvas, TextureQuery};
use std::time::Duration;

pub struct Display {
    running: bool,
    font_size: u16,
    pixel_size: i32,
    context: sdl2::Sdl,
    canvas: WindowCanvas,
}

fn get_graphics_driver() -> Option<u32> {
    for (index, item) in sdl2::render::drivers().enumerate() {
        if item.name == "opengl" {
            return Some(index as u32);
        }
    }
    return None;
}

impl Display {
    pub fn new() -> Display {
        let context = sdl2::init().unwrap();
        let video_subsystem = context.video().unwrap();

        let window = video_subsystem
                        .window("Digit visualizer", 280, 280)
                        .position_centered()
                        .opengl()
                        .build().unwrap();

        let canvas = window.into_canvas()
                           .index(get_graphics_driver().unwrap())
                           .build().unwrap();

        Display {
            running: true,
            canvas: canvas,
            font_size: 20,
            pixel_size: 10,
            context: context,
        }
    }

    fn draw_text(&mut self, font: &Font, text: &str, x: i32, y: i32) {
        let texture_creator = self.canvas.texture_creator();
        let surface = font.render(text)
                          .blended(Color::RGBA(255, 255, 255, 255))
                          .unwrap();
        let texture = texture_creator.create_texture_from_surface(&surface).unwrap();
        let TextureQuery {width, height, ..} = texture.query();

        let target = Rect::new(x, y, width, height);
        self.canvas.copy(&texture, None, Some(target)).unwrap();
    }

    fn draw_rect(&mut self, x: i32, y: i32, r: u8, g: u8, b: u8) {
        self.canvas.set_draw_color(Color::RGB(r,g,b));
        self.canvas.fill_rect(Rect::new(
            x * self.pixel_size,
            y * self.pixel_size,
            self.pixel_size as u32,
            self.pixel_size as u32,
        )).unwrap();
    }

    fn draw(&mut self, img_data: &Vec<f64>, l1: &u8, l2: &u8, font: &Font) {
        self.canvas.set_draw_color(Color::RGB(255, 255, 255));
        self.canvas.clear();

        let size: i32 = 28;
        for y in 0..size {
            for x in 0..size {
                let i = (y * size + x) as usize;
                let c = img_data[i] as u8;
                self.draw_rect(x,y,c,c,c);
            }
        }

        let s1 = format!("Label: {}", l1);
        let s2 = format!("Predicted: {}", l2);
        self.draw_text(font, &s1, 0, 0);
        self.draw_text(font, &s2, 0, self.font_size as i32);

        self.canvas.present();
    }

    pub fn render_digit(&mut self, imgs: Vec<Vec<f64>>, labels: Vec<u8>, predictions: Vec<u8>, img_max: i32) {
        let mut img_index: usize = 0;
        let mut events = self.context.event_pump().unwrap();
        let ttf = sdl2::ttf::init().unwrap();
        let font = ttf.load_font("data/Arial.ttf", self.font_size).unwrap();

        while self.running {
            for event in events.poll_iter() {
                match event {
                    Event::Quit {..} => self.running = false,
                    Event::KeyDown {keycode: Some(Keycode::Space), ..} => {
                        img_index += 1;
                        if img_index >= img_max as usize {
                            img_index = 0;
                        }
                    },
                    _ => {},
                }

                self.draw(&imgs[img_index], &labels[img_index], &predictions[img_index], &font);
                ::std::thread::sleep(Duration::new(0, 1000000000 / 60));
            }
        }
    }
}
