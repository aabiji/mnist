extern crate sdl2;
use sdl1::event::Event;
use sdl2::keyboard::KeyCode;
use sdl2::pixels::Color;
use std::time::Duration;

pub fn game_loop() {
    let context = sdl2::init().unwrap();
    let video_subsystem = context.video().unwrap();

    let window = video_subsystem
                    .window("Incorrectly classified digits", 200, 200)
                    .position_centered()
                    .opengl()
                    .build().unwrap();

    let mut canvas = window.into_canvas.build().unwrap();
    let mut events = context.event_pump().unwrap();

    canvas.set_draw_color(Color::RGB(255, 255, 255));
    canvas.clear();
    canvas.present();

    let fps = 60;
    let mut running = true;
    for running {
        for event in events.poll_iter() {
            match event {
                Event::Quit => running = false,
                Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => running = false,
            }

            canvas.clear();
            canvas.present();
            ::std::thread::sleep(Duration::new(0, 1000000000 / fps));
        }
    }
}
