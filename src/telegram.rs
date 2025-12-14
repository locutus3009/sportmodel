use chrono::Local;
use teloxide::{prelude::*, utils::command::BotCommands};

pub(crate) async fn start_bot() {
    let bot = Bot::from_env();

    Command::repl(bot, answer).await;
}

#[derive(BotCommands, Clone)]
#[command(
    rename_rule = "lowercase",
    description = "These commands are supported:"
)]
enum Command {
    #[command(description = "display this text.")]
    Help,
    #[command(description = "handle a body weight.")]
    Bodyweight(f32),
}

async fn answer(bot: Bot, msg: Message, cmd: Command) -> ResponseResult<()> {
    let today = Local::now().date_naive();
    match cmd {
        Command::Help => {
            bot.send_message(msg.chat.id, Command::descriptions().to_string())
                .await?
        }
        Command::Bodyweight(bodyweight) => {
            bot.send_message(
                msg.chat.id,
                format!("Your body weight for {:?} is {bodyweight}kg.", today),
            )
            .await?
        }
    };

    Ok(())
}
