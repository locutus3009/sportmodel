use std::path::PathBuf;
use std::sync::Arc;

use chrono::{Local, NaiveDate};
use teloxide::{
    dispatching::UpdateFilterExt,
    dptree,
    prelude::*,
    utils::command::{BotCommands, ParseError},
};
use umya_spreadsheet::*;

use crate::server::AppState;

pub(crate) async fn start_bot(state: Arc<AppState>) {
    let bot = Bot::from_env();

    Dispatcher::builder(
        bot,
        Update::filter_message()
            .branch(dptree::entry().filter_command::<Command>().endpoint(answer))
            .branch(dptree::filter(|_: Message| true).endpoint(handle_invalid_command)),
    )
    .dependencies(dptree::deps![state])
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
    NeckAndWaist(u32, u32),
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
    for i in 0..last_row {
        let date_cell = sheet.get_cell_mut((1, i)).get_value_number();
        let type_cell = sheet.get_cell_mut((4, i)).get_value();
        if date_cell == Some(date_excel) && type_cell == type_ {
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

    // Save
    writer::xlsx::write(&book, path)?;
    Ok(result)
}

async fn answer(bot: Bot, msg: Message, cmd: Command, state: Arc<AppState>) -> ResponseResult<()> {
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
                    append_excel(&state.file_path, today, calories as f64, None, "calories")
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

async fn handle_invalid_command(bot: Bot, msg: Message) -> ResponseResult<()> {
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
