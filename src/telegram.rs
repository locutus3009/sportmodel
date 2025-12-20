use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use chrono::{Duration, Local, NaiveDate};
use teloxide::{
    dispatching::UpdateFilterExt,
    dptree,
    prelude::*,
    types::InputFile,
    utils::command::{BotCommands, ParseError},
};
use umya_spreadsheet::*;

use crate::domain::Movement;
use crate::server::{AppState, WsMessage};

use image::{ImageBuffer, ImageOutputFormat, Rgb}; // Add these imports
use plotters::prelude::*;
use plotters::style::Color;
use std::io::Cursor;

/// Checks if a user is authorized based on the whitelist.
/// Returns true if authorized, false otherwise.
/// Logs all authorization attempts.
fn is_authorized(user_id: UserId, allowed: &[i64]) -> bool {
    let id = user_id.0 as i64;

    if allowed.is_empty() {
        // Discovery mode: log and deny
        log::warn!(
            "Telegram user_id={} DENIED - add to TELEGRAM_ALLOWED_USERS to allow",
            id
        );
        false
    } else if allowed.contains(&id) {
        log::info!("Telegram user_id={} authorized", id);
        true
    } else {
        log::warn!(
            "Telegram user_id={} DENIED - add to TELEGRAM_ALLOWED_USERS to allow",
            id
        );
        false
    }
}

pub(crate) async fn start_bot(state: Arc<AppState>) {
    let bot = Bot::from_env();

    Dispatcher::builder(
        bot,
        Update::filter_message()
            .branch(dptree::entry().filter_command::<Command>().endpoint(answer))
            .branch(dptree::filter(|_: Message| true).endpoint(handle_invalid_command)),
    )
    .dependencies(dptree::deps![state.clone()])
    .build()
    .dispatch()
    .await;
}

#[derive(BotCommands, Clone)]
#[command(
    rename_rule = "lowercase",
    description = "These commands are supported:"
)]
enum Command {
    #[command(description = "display this text.")]
    Help,
    #[command(description = "get daily calorie consumption.")]
    Tdee,
    #[command(description = "[weight] handle body weight.")]
    Bodyweight(f64),
    #[command(
        description = "[weight] [reps] handle back squat.",
        parse_with = "split"
    )]
    Squat(f64, u32),
    #[command(
        description = "[weight] [reps] handle bench press.",
        parse_with = "split"
    )]
    Bench(f64, u32),
    #[command(description = "[weight] [reps] handle deadlift.", parse_with = "split")]
    Deadlift(f64, u32),
    #[command(description = "[weight] [reps] handle snatch.", parse_with = "split")]
    Snatch(f64, u32),
    #[command(
        description = "[weight] [reps] handle clean & jerk.",
        parse_with = "split"
    )]
    Cj(f64, u32),
    #[command(description = "[calories] handle calories.")]
    Calories(u32),
    #[command(
        description = "[neck] [waist] handle neck and waist.",
        parse_with = "split"
    )]
    NeckAndWaist(f64, f64),
}

fn date_to_excel_serial(date: NaiveDate) -> f64 {
    // Excel epoch is 1899-12-30 (accounting for Excel's leap year bug)
    let excel_epoch = NaiveDate::from_ymd_opt(1899, 12, 30).unwrap();
    (date - excel_epoch).num_days() as f64
}

fn append_excel(
    path: &PathBuf,
    date: NaiveDate,
    weight: f64,
    repetitions: Option<u32>,
    type_: &str,
) -> Result<&'static str, Box<dyn std::error::Error>> {
    // Open existing file
    let mut book = reader::xlsx::read(path)?;
    let sheet = book.get_sheet_mut(&0).ok_or("Sheet not found")?;
    let date_excel = date_to_excel_serial(date);
    let mut result = "";

    // Find next empty row or existing row with the same date and type
    let last_row = sheet.get_highest_row() + 1;
    let mut update_row = last_row;

    // Search for existing row - use get_cell (non-mut) to avoid creating phantom cells
    for i in 0..last_row {
        // Try non-mutating access first - only creates cell if it exists
        let has_date = if let Some(cell) = sheet.get_cell((1, i)) {
            cell.get_value_number() == Some(date_excel)
        } else {
            false
        };

        let has_matching_type = if let Some(cell) = sheet.get_cell((4, i)) {
            cell.get_value() == type_
        } else {
            false
        };

        if has_date && has_matching_type {
            update_row = i;
            result = " [updated]";
            break;
        }
    }

    // Write cells
    let date_cell = sheet.get_cell_mut((1, update_row));
    date_cell.set_value_number(date_excel);
    date_cell
        .get_style_mut()
        .get_number_format_mut()
        .set_format_code("yyyy-mm-dd");

    sheet.get_cell_mut((2, update_row)).set_value_number(weight);
    if let Some(reps) = repetitions {
        sheet.get_cell_mut((3, update_row)).set_value_number(reps);
    }
    sheet.get_cell_mut((4, update_row)).set_value(type_);

    // Save and explicitly flush to disk
    writer::xlsx::write(&book, path)?;
    drop(book); // Explicit drop to release file handle

    // Force OS to flush by opening and syncing
    log::debug!("Syncing Excel file to disk: {}", path.display());
    let file = File::open(path)?;
    file.sync_all()?; // Guarantees data and metadata synced to disk
    drop(file);

    // Small delay to ensure file watcher sees complete file
    thread::sleep(std::time::Duration::from_millis(50));
    log::debug!("Excel file sync completed");

    Ok(result)
}

async fn create_plot(
    state: Arc<AppState>,
    movement: Movement,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let today = Local::now().date_naive();
    let start_date = today - Duration::days(180);
    let end_date = today + Duration::days(90);

    let data = state.data.read().await;
    let (observations, predictions) = {
        let raw_obs = data
            .training_data
            .get(movement)
            .map(|points| {
                points
                    .iter()
                    .filter(|p| p.date >= start_date && p.date <= today)
                    .map(|p| (p.date, p.value))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let preds = data
            .analyses
            .get(&movement)
            .map(|analysis| {
                analysis
                    .predictions
                    .iter()
                    .filter(|p| p.date >= start_date && p.date <= end_date)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        (raw_obs, preds)
    };

    const HEIGHT: usize = 600;
    const WIDTH: usize = 1000;
    let mut plot_data = vec![0; HEIGHT * WIDTH * 3];

    {
        let root =
            BitMapBackend::with_buffer(&mut plot_data, (WIDTH.try_into()?, HEIGHT.try_into()?))
                .into_drawing_area();

        let bg_color = RGBColor(30, 30, 30);
        let text_color = RGBColor(176, 176, 176);
        let accent_color = RGBColor(33, 150, 243);
        let success_color = RGBColor(76, 175, 80);
        let grid_color = RGBColor(61, 61, 61);

        root.fill(&bg_color)?;

        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;

        for &(_, val) in &observations {
            min_y = min_y.min(val);
            max_y = max_y.max(val);
        }
        for pred in &predictions {
            min_y = min_y.min(pred.mean);
            max_y = max_y.max(pred.mean);
        }

        if min_y == f64::MAX {
            min_y = 0.0;
            max_y = 100.0;
        } else {
            let padding = (max_y - min_y) * 0.1;
            min_y = (min_y - padding).floor();
            max_y = (max_y + padding).ceil();
        }

        // Вычисляем количество меток Y для шага в 1кг
        let y_label_count = ((max_y - min_y).abs() as usize).max(1);

        let mut chart = ChartBuilder::on(&root)
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(start_date..end_date, min_y..max_y)?;

        chart
            .configure_mesh()
            .bold_line_style(grid_color.stroke_width(1))
            .light_line_style(grid_color.stroke_width(1))
            .axis_style(grid_color.stroke_width(1))
            .label_style(("sans-serif", 14).into_font().color(&text_color))
            // Настройка сетки X: каждые 10 дней
            .x_labels(20)
            .x_label_formatter(&|d| d.format("%d.%m").to_string())
            // Настройка сетки Y: ровно по 1кг
            .y_labels(y_label_count)
            .draw()?;

        // Отрисовка CI Polygon
        let mut ci_polygon = Vec::new();
        for p in &predictions {
            ci_polygon.push((p.date, p.ci_upper()));
        }
        for p in predictions.iter().rev() {
            ci_polygon.push((p.date, p.ci_lower()));
        }

        if !ci_polygon.is_empty() {
            chart.draw_series(std::iter::once(Polygon::new(
                ci_polygon,
                accent_color.mix(0.15).filled(),
            )))?;
        }

        // Линия тренда
        chart.draw_series(LineSeries::new(
            predictions.iter().map(|p| (p.date, p.mean)),
            accent_color.stroke_width(2),
        ))?;

        // Точки данных
        chart.draw_series(PointSeries::of_element(
            observations,
            4,
            ShapeStyle::from(&success_color).filled(),
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))?;

        // Вертикальная линия "Сегодня"
        chart.draw_series(LineSeries::new(
            vec![(today, min_y), (today, max_y)],
            text_color.mix(0.8).stroke_width(2),
        ))?;

        root.present()?;
    }

    let img_buffer: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(WIDTH as u32, HEIGHT as u32, plot_data)
            .ok_or("Failed to create image buffer")?;

    let mut png_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut png_bytes);
    img_buffer.write_to(&mut cursor, ImageOutputFormat::Png)?;

    Ok(png_bytes)
}

/// Spawns an async task to wait for data reload and send chart.
///
/// This function waits for DataUpdated event on the provided receiver,
/// then generates and sends a chart for the specified movement.
fn spawn_chart_sender(bot: Bot, state: Arc<AppState>, chat_id: ChatId, movement: Movement) {
    let mut rx = state.ws_broadcast.subscribe();

    tokio::spawn(async move {
        // Wait for DataUpdated with timeout
        let timeout_duration = std::time::Duration::from_secs(5);
        let result = tokio::time::timeout(timeout_duration, async {
            while let Ok(msg) = rx.recv().await {
                if matches!(msg, WsMessage::DataUpdated) {
                    drop(rx);
                    return true;
                }
            }
            log::warn!("Broadcast loop ended without DataUpdated");
            false
        })
        .await;

        match result {
            Ok(true) => {
                log::info!("Data reload completed, generating chart for {:?}", movement);

                match create_plot(state, movement).await {
                    Ok(plot) => {
                        let send_result = bot
                            .send_photo(
                                chat_id,
                                InputFile::memory(plot).file_name(format!(
                                    "{:?}_{:?}.png",
                                    Local::now().date_naive(),
                                    movement
                                )),
                            )
                            .await;

                        if let Err(e) = send_result {
                            log::error!("Failed to send chart to Telegram: {:?}", e);
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to generate plot: {:?}", e);
                    }
                }
            }
            Ok(false) => {
                log::error!("Broadcast channel closed while waiting for data reload");
            }
            Err(_) => {
                log::error!("Timeout waiting for data reload for {:?} chart", movement);
            }
        }
    });
}

async fn answer(bot: Bot, msg: Message, cmd: Command, state: Arc<AppState>) -> ResponseResult<()> {
    // Authorization check
    let authorized = msg.from.as_ref().map_or(false, |user| {
        is_authorized(user.id, &state.telegram_allowed_users)
    });

    if !authorized {
        bot.send_message(msg.chat.id, "⚠️ Access denied. Contact bot administrator.")
            .await?;
        return Ok(());
    }

    let today = Local::now().date_naive();
    match cmd {
        Command::Help => {
            bot.send_message(msg.chat.id, Command::descriptions().to_string())
                .await?
        }
        Command::Tdee => {
            let data = state.data.read().await;

            let tdee_info = match &data.tdee {
                Ok(t) => format!(
                    "Data for last 28 days:\nAverage TDEE: {:.0} kcal\nToday TDEE: {:.0} kcal\nAverage intake: {:.0}\nWeight change: {:.1}kg",
                    t.average_tdee, t.tdee, t.avg_calories, t.weight_change_kg
                ),
                Err(_) => "TDEE: unknown".to_string(),
            };
            drop(data);

            bot.send_message(msg.chat.id, tdee_info).await?
        }
        Command::Bodyweight(bodyweight) => {
            spawn_chart_sender(
                bot.clone(),
                Arc::clone(&state),
                msg.chat.id,
                Movement::Bodyweight,
            );

            bot.send_message(
                msg.chat.id,
                format!(
                    "Your body weight for {:?} is {bodyweight}kg.{}",
                    today,
                    append_excel(&state.file_path, today, bodyweight, None, "bodyweight").unwrap()
                ),
            )
            .await?
        }
        Command::Squat(weight, reps) => {
            spawn_chart_sender(
                bot.clone(),
                Arc::clone(&state),
                msg.chat.id,
                Movement::Squat,
            );

            bot.send_message(
                msg.chat.id,
                format!(
                    "Your back squat for {:?} is {weight}kg x {reps}.{}",
                    today,
                    append_excel(&state.file_path, today, weight, Some(reps), "squat").unwrap()
                ),
            )
            .await?
        }
        Command::Bench(weight, reps) => {
            spawn_chart_sender(
                bot.clone(),
                Arc::clone(&state),
                msg.chat.id,
                Movement::Bench,
            );

            bot.send_message(
                msg.chat.id,
                format!(
                    "Your bench press for {:?} is {weight}kg x {reps}.{}",
                    today,
                    append_excel(&state.file_path, today, weight, Some(reps), "bench").unwrap()
                ),
            )
            .await?
        }
        Command::Deadlift(weight, reps) => {
            spawn_chart_sender(
                bot.clone(),
                Arc::clone(&state),
                msg.chat.id,
                Movement::Deadlift,
            );

            bot.send_message(
                msg.chat.id,
                format!(
                    "Your deadlift for {:?} is {weight}kg x {reps}.{}",
                    today,
                    append_excel(&state.file_path, today, weight, Some(reps), "deadlift").unwrap()
                ),
            )
            .await?
        }
        Command::Snatch(weight, reps) => {
            spawn_chart_sender(
                bot.clone(),
                Arc::clone(&state),
                msg.chat.id,
                Movement::Snatch,
            );

            bot.send_message(
                msg.chat.id,
                format!(
                    "Your snatch for {:?} is {weight}kg x {reps}.{}",
                    today,
                    append_excel(&state.file_path, today, weight, Some(reps), "snatch").unwrap()
                ),
            )
            .await?
        }
        Command::Cj(weight, reps) => {
            // Spawn task to send chart after data reload
            spawn_chart_sender(
                bot.clone(),
                Arc::clone(&state),
                msg.chat.id,
                Movement::CleanAndJerk,
            );

            bot.send_message(
                msg.chat.id,
                format!(
                    "Your clean & jerk for {:?} is {weight}kg x {reps}.{}",
                    today,
                    append_excel(&state.file_path, today, weight, Some(reps), "cj").unwrap()
                ),
            )
            .await?
        }
        Command::Calories(calories) => {
            bot.send_message(
                msg.chat.id,
                format!(
                    "Your calories for {:?} is {calories}kcal.{}",
                    today,
                    append_excel(&state.file_path, today, calories as f64, None, "calorie")
                        .unwrap()
                ),
            )
            .await?
        }
        Command::NeckAndWaist(neck, waist) => {
            let updated1 =
                append_excel(&state.file_path, today, neck as f64, None, "neck").unwrap();
            let updated2 =
                append_excel(&state.file_path, today, waist as f64, None, "waist").unwrap();
            bot.send_message(
                msg.chat.id,
                format!(
                    "Your neck for {:?} is {neck}cm{updated1}, waist is {waist}cm{updated2}.",
                    today
                ),
            )
            .await?
        }
    };

    Ok(())
}

async fn handle_invalid_command(
    bot: Bot,
    msg: Message,
    state: Arc<AppState>,
) -> ResponseResult<()> {
    // Authorization check
    let authorized = msg.from.as_ref().map_or(false, |user| {
        is_authorized(user.id, &state.telegram_allowed_users)
    });

    if !authorized {
        bot.send_message(msg.chat.id, "⚠️ Access denied. Contact bot administrator.")
            .await?;
        return Ok(());
    }

    let text = msg.text().unwrap_or("");

    // Try to parse and get the actual error
    match Command::parse(text, "") {
        Ok(_) => {} // Shouldn't happen, but ignore
        Err(err) => {
            let error_msg = match err {
                ParseError::TooFewArguments {
                    expected,
                    found,
                    message,
                } => {
                    format!("Missing argument. Expected {expected}, got {found}.\nUsage: {message}")
                }
                ParseError::TooManyArguments {
                    expected,
                    found,
                    message,
                } => {
                    format!(
                        "Too many arguments. Expected {expected}, got {found}.\nUsage: {message}"
                    )
                }
                ParseError::IncorrectFormat(err) => {
                    format!("Invalid format: {err}")
                }
                ParseError::UnknownCommand(cmd) => {
                    format!("Unknown command: {cmd}\n\n{}", Command::descriptions())
                }
                ParseError::WrongBotName(_) => return Ok(()),
                ParseError::Custom(err) => format!("Error: {err}"),
            };

            bot.send_message(msg.chat.id, error_msg).await?;
        }
    }

    Ok(())
}
