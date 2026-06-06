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
    LoadCommentsFromCSV "p:\ollama-dev\comments.csv"

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

Sub LoadCommentsFromCSV(filePath As String)
    Dim fileNum As Integer
    Dim textLine As String
    Dim comment As String
    Dim i As Long
    Dim j As Integer
    Dim reservoir(1 To 100) As String
    Dim reservoirCount As Integer

    ' Initialize random number generator
    Randomize

    On Error Resume Next
    fileNum = FreeFile
    Open filePath For Input As #fileNum

    i = 0
    reservoirCount = 0

    ' Reservoir sampling algorithm to get 100 random samples
    Do While Not EOF(fileNum)
        Line Input #fileNum, textLine
        i = i + 1

        ' Extract comment part (before the first |)
        comment = Split(textLine, "|")(0)
        comment = Trim(comment)

        ' Only process non-empty comments
        If Len(comment) > 0 Then
            If reservoirCount < 100 Then
                ' Fill reservoir
                reservoirCount = reservoirCount + 1
                reservoir(reservoirCount) = comment
            Else
                ' Replace elements with gradually decreasing probability
                j = Int((i * Rnd) + 1)
                If j <= 100 Then
                    reservoir(j) = comment
                End If
            End If
        End If
    Loop

    Close #fileNum

    ' Transfer from reservoir to comments collection
    For j = 1 To reservoirCount
        comments.Add reservoir(j)
    Next j

    On Error GoTo 0

    ' If no comments were loaded, add some sample text
    If comments.Count = 0 Then
        comments.Add "Sample comment 1"
        comments.Add "Sample comment 2"
        comments.Add "Sample comment 3"
        comments.Add "Sample comment 4"
        comments.Add "Sample comment 5"
    End If
End Sub

Sub StopEnhancedEffect()
    isEffectRunning = False
    On Error Resume Next
    Application.OnTime Now + TimeValue("00:00:01"), "UpdateEnhancedEffect", , False
    On Error GoTo 0
End Sub