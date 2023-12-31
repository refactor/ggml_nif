-module(logger_color_formatter).

-export([check_config/1, format/2]).

-define(DEFAULT_FORMATTER, logger_formatter).

check_config(Config) ->
    _Formatter = maps:get(formatter, Config, logger_formatter),
    _Colors = maps:get(colors, Config, #{}),
    ?DEFAULT_FORMATTER:check_config(update_config(os:getenv("NO_COLOR"), Config, debug)).

format(LogEvent = #{level := Level}, Config) ->
    Config2 = update_config(os:getenv("NO_COLOR"), Config, Level),
    ?DEFAULT_FORMATTER:format(LogEvent, Config2).

update_config(_NoColor = false,
              Config0 = #{template := Template, colors := Colors},
              Level) ->
    Config = maps:remove(colors, Config0),
    Config#{legacy_header => false,
            single_line => true,
            template => update_template(Template, maps:merge(default_colors(), Colors), Level)};
update_config(_NoColor = false, Config = #{template := Template}, Level) ->
    Config#{legacy_header => false,
            single_line => true,
            template => update_template(Template, default_colors(), Level)};
update_config(_NoColor, Config, _Level) ->
    Config#{legacy_header => false, single_line => true}.

update_template(Template, Colors, Level) ->
    lists:map(fun (color) ->
                      maps:get(Level, Colors);
                  (reset) ->
                      maps:get(reset, Colors);
                  (Item) ->
                      Item
              end,
              Template).

default_colors() ->
    #{debug => "\e[0;38m",
      info => "\e[1;37m",
      notice => "\e[1;36m",
      warning => "\e[1;33m",
      error => "\e[1;31m",
      critical => "\e[1;35m",
      alert => "\e[1;44m",
      emergency => "\e[1;41m",
      % Not a colour, but this'll do.
      reset => "\e[0m"}.
