from statbelt import main


def test_cli_smoke(capsys) -> None:
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello from statbelt!"
    assert captured.err == ""
