CC := clang
CFLAGS := -O0 -Wall -g
TARGETS := pizzashop numbergame password accounts

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

%: %.c
	@echo Compiling $<
	@$(CC) $(CFLAGS) -o $@ $< 2> /dev/null
