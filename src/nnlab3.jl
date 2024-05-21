module nnlab3

using Telegram, Telegram.API
using HTTP
using Logging, LoggingExtras
using Images
using Printf
using Base.Threads

include("predict.jl")

function msg_route(msg)
    if :photo in keys(msg.message)
        maxres_id = msg.message.photo[findmax(x -> x.width * x.height, msg.message.photo)[2]].file_id
        file_ref = getFile(file_id=maxres_id)
        url = "https://api.telegram.org/file/bot" * ENV["BOT_TOKEN"] * "/" * file_ref.file_path
        img = HTTP.get(url).body |> IOBuffer |> Images.load
        prepared_img = reshape(Float32.(imresize(img, (128, 128)) |> channelview), 128, 128, 3) |> collect

        function report_prediction(prepared_img::Array{Float32, 3})
            prediction = predictor.predict(prepared_img)
            decision = Nothing
            if prediction[1] > 0.7
                decision = " daisy"
            elseif prediction[1] < 0.3
                decision = " dandelion"
            else
                decision = "... actually, what is that?"
            end
            text = (@sprintf "That's a%s!" decision)
            @debug text
            sendMessage(text=text, chat_id = msg.message.chat.id)

        end
        @spawn report_prediction(prepared_img)
    else
        sendMessage(text = "Photo 404, what am I supposed to recognize??", chat_id = msg.message.chat.id)
    end
end

function julia_main()::Cint
    predictor.init(ENV["MODEL_PATH"])
    tg = TelegramClient(ENV["BOT_TOKEN"])
    println("ready to serve!")
    try
        run_bot() do msg
            msg_route(msg)
        end
    catch e
        if isa(e, InterruptException)
            return 0
        else
            println(e)
        end
    end

    return 0
end

end
