Attribute VB_Name = "EnhancedSymbolEffect"
Option Explicit

Dim startTime As Double
Dim isEffectRunning As Boolean
Dim comments As Collection
Dim commentIndex As Integer
Dim symbolColumns As Collection

Sub StartEnhancedSymbolEffect()
    ' Initialize variables
    isEffectRunning = True
    Set comments = New Collection
    Set symbolColumns = New Collection
    commentIndex = 1

    ' Initialize columns with random speeds
    Dim i As Integer
    For i = 1 To 10 ' Work with first 10 columns
        symbolColumns.Add Int((5 * Rnd) + 1) ' Speed between 1-5
    Next i

    ' Load comments from CSV
    LoadCommentsFromCSV

    ' Start effect
    startTime = Timer
    Cells.Clear

    ' Start the animation
    Application.OnTime Now + TimeValue("00:00:01"), "UpdateEnhancedEffect"
End Sub

Sub UpdateEnhancedEffect()
    If Not isEffectRunning Then Exit Sub

    Dim elapsedSeconds As Integer
    elapsedSeconds = Timer - startTime

    ' First 3 seconds: scrolling symbols
    If elapsedSeconds < 3 Then
        ScrollSymbolsAsync
        Application.OnTime Now + TimeValue("00:00:01"), "UpdateEnhancedEffect"
    ' After 10 seconds: replace with comments
    ElseIf elapsedSeconds >= 10 And elapsedSeconds < 20 Then
        ReplaceWithComments
        Application.OnTime Now + TimeValue("00:00:01"), "UpdateEnhancedEffect"
    ' Stop after 20 seconds
    Else
        StopEnhancedEffect
    End If
End Sub

Sub ScrollSymbolsAsync()
    Dim col As Integer
    Dim speed As Integer
    Dim i As Integer
    Dim cellValue As String
    Dim count As Integer

    ' For each column, scroll at its own speed
    For col = 1 To symbolColumns.Count
        speed = symbolColumns(col)

        ' Move symbols up by 'speed' rows
        For i = 1 To Rows.Count - speed
            If Cells(i + speed, col).Value <> "" Then
                Cells(i, col).Value = Cells(i + speed, col).Value
            Else
                Cells(i, col).Value = ""
            End If
        Next i

        ' Add new symbols at the bottom
        For i = Rows.Count - speed + 1 To Rows.Count
            ' Random number of backticks (0-300)
            count = Int((300 * Rnd))
            cellValue = String(count, "`")
            Cells(i, col).Value = cellValue
        Next i
    Next col
End Sub

Sub ReplaceWithComments()
    Dim i As Integer, j As Integer
    Dim rand As Integer
    Dim comment As String

    ' Replace random symbols with comments moving from top to bottom
    For i = 1 To Rows.Count
        For j = 1 To symbolColumns.Count
            ' Only replace if cell contains symbols
            If Cells(i, j).Value Like "*`*" Then
                ' Randomly decide whether to replace (higher chance at top)
                rand = Int((10 * Rnd) + 1)
                Dim replaceChance As Integer
                replaceChance = Application.WorksheetFunction.Max(1, 8 - (i \ 10)) ' Higher chance at top

                If rand <= replaceChance Then
                    ' Get next comment
                    If commentIndex <= comments.Count Then
                        comment = comments(commentIndex)
                        commentIndex = commentIndex + 1
                        If commentIndex > comments.Count Then commentIndex = 1 ' Loop back
                        Cells(i, j).Value = comment
                    End If
                End If
            End If
        Next j
        ' Small delay effect for top-to-bottom movement
        If i Mod 3 = 0 Then DoEvents
    Next i
End Sub

Sub LoadCommentsFromCSV()
    comments.Add "а путин просто ссыкун вместе со своими шавками боитса конкурентов которые видят реалии яснее чем он"
    comments.Add "колян молодец"
    comments.Add "так я быстрее уеду если вы успокоите [психушка]"
    comments.Add "петух молчит, но столько фоток поставил странно, может его убрали"
    comments.Add "пошли нахуй эти люди, какоето больное быдло, я уже жалко что я русский, реально посмотреть на них не, нигде не снимаюс щаз"
    comments.Add "вес обнаружен, но датчики почемуто говорят 0%"
    comments.Add "он все спускалса и спускал с а все ниже"
    comments.Add "вот это норм язык, советую детям для изучения."
    comments.Add "впереди, ну так это 2020 инфа"
    comments.Add "нет, не время. время загружать"
    comments.Add "информация анализируеца ..."
    comments.Add "вот это норм шутки которые вы чередуете для привлечения внимания только это инфа из прошлых лет а вы не умеете натягивать новое на все старое сразу поэтому постояно забанены"
    comments.Add "пиздец крыса такая"
    comments.Add "обе"
    comments.Add "сникерс с фундуком и спецназовцами"
    comments.Add "ну да, точно про меня. ну ты же знаеш"
    comments.Add "пушо хуйня платья, вон бля на пинтересте антоха показывал вчера норм платья"
    comments.Add "это уже проходили"
    comments.Add "мне поебать если чесно, пускай дальше хуи разглядывает на столе и картинах, мы"
    comments.Add "это вопрос"
    comments.Add "медсистрской"
    comments.Add "путин в машинном обучении» - бесплатный курс, который"
    comments.Add "пушо они абнормальные все, особенно в росси и дагестанских странах"
    comments.Add "так и записал, буду повторять на ночь и передам детям"
    comments.Add "асфклонски"
    comments.Add "теже прогеры только гуманитарий"
    comments.Add "не"
    comments.Add "особенно когда это не сходиться с реальностью и ты об этом знаеш"
    comments.Add "э"
    comments.Add "пост хттп/1.1"
    comments.Add "звучит как бред конечно, но поверить можно. а что бы было если бы такая хуйня случилас когда небыло людей?"
    comments.Add "я всегда балансирую. нахуй постояно укрытым"
    comments.Add "и ко мне в кроватку"
    comments.Add "типо у автора есть точка зрения закона"
    comments.Add "конешно"
    comments.Add "cексуашка""
    comments.Add "вы чото попутали или коры поели"
    comments.Add "игры с ведром"
    comments.Add "и там управляет жириновски"
    comments.Add "ну ты да""
    comments.Add "ну а я нет""
    comments.Add "за батоном в народный, я в качалке - немогу разговаривать""
    comments.Add "чья?""
    comments.Add "отозвать антона"
    comments.Add "обучение это не бизнесс задачи, бизнесс использует. навыком можно заработать обучая"
    comments.Add "смарика, коменты не закрыты о чем это может говорить?"
    comments.Add "подглядывать некультурно"
    comments.Add "какая?"
    comments.Add "live из улья"
    comments.Add "локальная чпокерия установлена обнаружена глобальная упячка"
    comments.Add "можно просто обдолбаца и установить канал пивные проходят к своим носкам"
    comments.Add "тот самый саентолог!"
    comments.Add "и чо"
    comments.Add "хуйня"
    comments.Add "все знают что никто никого не ремувнул. все мувнулись из специальности в операторов дэпэтэ"
    comments.Add "бля этот сплоит еще в ультре в 2010 проходили и обосались"
    comments.Add "это дефейс?"
    comments.Add "ночью в 3"
    comments.Add "политех"
    comments.Add "пошли нахуй от неё"
    comments.Add "я ща радиоэфир буду модифицировать. посморим чо туд младенцы деф яйсы умеют"
    comments.Add "мощные агенты принца путина выполняют контртеррристические задачи, под руководством верхновной обезьяны"
    comments.Add "если будете дальше кормить меня своими ""де фейсами"" я буду жоско лолировать над вами"
    comments.Add "послано байт: 0 послано обезьян: 11"
    comments.Add "сукии давления недостаточно младенцы ебаные"
    comments.Add "механики, иконы ..."
    comments.Add "Возразить,""
    comments.Add "для пидор√""
    comments.Add "расходимся он его там создал"
    comments.Add "тока ваш антифрод на ифах сделан, а тот коунтер террор основанный на рандоме, который вы обосрали не поняв сути 10 лет назад работал атомарно"
    comments.Add "приехал пожрал написал в телегу рядом в комнату уехал"
    comments.Add "да хуй тебе я зиму в кьюеуее поставил три раза""
    comments.Add "а я чо аналитик ебаный над твоими вопросами думать""
    comments.Add "заменить на пнор""
    comments.Add "чем я и занимаюсь с ночи до утра"
    comments.Add "не, у них все под контролем"
    comments.Add "циник-пиздабол ебаный"
    comments.Add "выжимка кохагена"
    comments.Add "задача 2020-2025: найти пхпшника не долбоеба"
    comments.Add "через секс.ком"
    comments.Add "килтарианцы"
    comments.Add "потом перехват очередной раненой питерской камеры"
    comments.Add "роняю ей каммент видео текста"
    comments.Add "нюоз"
    comments.Add "апядь новоз"
    comments.Add "у контактовских статус понижен"
    comments.Add "терминатор пошатал ся по городу"
    comments.Add "акинфееф забил релейных"
    comments.Add "вы эт. не грустите ток."
    comments.Add "жужик"
    comments.Add "да им похуй"
    comments.Add "у него порт оп ен""
    comments.Add "охранег холодетц""
    comments.Add "он мимикрировал под грунт""
    comments.Add "гос кноттинги спиздили прям из счетной палаты"
    comments.Add "В час ночи на Гороховой улице 19-летний десантник Георгий Сударушкин, прогуливаясь с друзьями захотел высадить с а"
    comments.Add "пешесса""
    comments.Add "оскорбительный""
    comments.Add "п о п рав"
    comments.Add "у hal""
End Sub

Sub StopEnhancedEffect()
    isEffectRunning = False
    On Error Resume Next
    Application.OnTime Now + TimeValue("00:00:01"), "UpdateEnhancedEffect", , False
    On Error GoTo 0
End Sub