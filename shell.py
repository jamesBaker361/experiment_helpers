#!/usr/bin/env python3
import paramiko
import cmd
import os
import sys
import readline  # arrow key history
import getpass
import fnmatch
import glob

import sys
import tty
import termios
import argparse

def input_password(prompt="Password: "):
    """Prompt for a password and display asterisks (*) as the user types"""
    print(prompt, end='', flush=True)
    password = ""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ('\n', '\r'):
                print("")  # newline
                break
            elif ch == '\x7f':  # backspace
                if len(password) > 0:
                    password = password[:-1]
                    # move cursor back, overwrite *, move cursor back
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
            else:
                password += ch
                sys.stdout.write('*')
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return password


class SFTPShell(cmd.Cmd):
    prompt = "sftp> "

    def __init__(self, hostname, username, key_filename=None, port=22):
        super().__init__()
        self.hostname = hostname
        self.username = username
        self.port = port
        self.key_filename = key_filename
        self.transport = None
        self.sftp = None
        self.connect()

    # Duo keyboard-interactive handler
    def duo_handler(self, title, instructions, prompt_list):
        responses = []
        for prompt, show_input in prompt_list:
            print(prompt,)  # print prompt
            resp = getpass.getpass(prompt)
            #resp = input("> ")      # type '1' to approve Duo push
            responses.append(resp)
        return responses

    def connect(self):
        print(f"Connecting to {self.hostname}...")
        self.transport = paramiko.Transport((self.hostname, self.port))
        self.transport.start_client()

        # Load SSH key if provided
        key = None
        if self.key_filename:
            key = paramiko.RSAKey.from_private_key_file(self.key_filename)

        # Keyboard-interactive for Duo MFA
        if not self.transport.is_authenticated():
            self.transport.auth_interactive(self.username, self.duo_handler)

        if not self.transport.is_authenticated():
            print("Authentication failed.")
            sys.exit(1)

        self.sftp = paramiko.SFTPClient.from_transport(self.transport)
        print("Connected successfully!")

    # -------------------
    # Basic SFTP commands
    # -------------------
    def do_ls(self, arg):
        path = arg or "."
        try:
            for f in self.sftp.listdir(path):
                print(f)
        except FileNotFoundError:
            print(f"{path} not found.")

    def do_cd(self, path):
        if path:
            try:
                self.sftp.chdir(path)
                print(f"Changed directory to {self.sftp.getcwd()}")
            except IOError:
                print(f"Directory {path} does not exist.")
        else:
            print("Usage: cd <remote_path>")

    def do_pwd(self, arg):
        print(self.sftp.getcwd())

    # -------------------
    # Recursive get
    # -------------------
    def do_get(self, arg):
        """get [-r] remote_file_or_dir_or_pattern"""
        parts = arg.split()
        recursive = False
        path = None
        if len(parts) == 2 and parts[0] == "-r":
            recursive = True
            path = parts[1]
        elif len(parts) == 1:
            path = parts[0]
        else:
            print("Usage: get [-r] remote_path_or_pattern")
            return

        if "*" in path or "?" in path:
            # Wildcard handling
            matches = [f for f in self.sftp.listdir(".") if fnmatch.fnmatch(f, path)]
            if not matches:
                print(f"No matches found for {path}")
                return
            for f in matches:
                if recursive and self._is_remote_dir(f):
                    self._get_recursive(f, f)
                else:
                    self.sftp.get(f, f)
                    print(f"Downloaded {f}")
        else:
            try:
                # No wildcard
                if recursive:
                    local_dir = os.path.basename(path.rstrip("/"))
                    os.makedirs(local_dir,exist_ok=True)
                    self._get_recursive(path, local_dir)
                else:
                        self.sftp.get(path, os.path.basename(path))
                        print(f"Downloaded {path}")
            except FileNotFoundError:
                print(f" {path} does not exist")

    def _get_recursive(self, remote_path, local_path):
        os.makedirs(local_path, exist_ok=True)
        for f in self.sftp.listdir(remote_path):
            remote_file = remote_path + "/" + f
            local_file = os.path.join(local_path, f)
            if self._is_remote_dir(remote_file):
                self._get_recursive(remote_file, local_file)
            else:
                self.sftp.get(remote_file, local_file)
                print(f"Downloaded {remote_file}")

    def _is_remote_dir(self, path):
        try:
            return paramiko.SFTPAttributes.from_stat(self.sftp.stat(path)).st_mode & 0o40000 == 0o40000
        except IOError:
            return False

    # -------------------
    # Recursive put
    # -------------------
    def do_put(self, arg):
        """put [-r] local_file_or_dir_or_pattern"""
        parts = arg.split()
        recursive = False
        path = None
        if len(parts) == 2 and parts[0] == "-r":
            recursive = True
            path = parts[1]
        elif len(parts) == 1:
            path = parts[0]
        else:
            print("Usage: put [-r] local_path_or_pattern")
            return

        # Wildcard handling
        if "*" in path or "?" in path:
            matches = glob.glob(path)
            if not matches:
                print(f"No matches found for {path}")
                return
            for f in matches:
                if recursive and os.path.isdir(f):
                    self._put_recursive(f, os.path.basename(f))
                elif os.path.isdir(f) is False:
                    self.sftp.put(f, os.path.basename(f))
                    print(f"Uploaded {f}")
        else:
            # No wildcard
            if recursive:
                self._put_recursive(path, os.path.basename(path))
            else:
                if os.path.exists(path):
                    if os.path.isdir(path) is False:
                        self.sftp.put(path, os.path.basename(path))
                        print(f"Uploaded {path}")
                else:
                    print(f"Local file {path} does not exist.")

    def _put_recursive(self, local_path, remote_path):
        try:
            self._mkdir_remote(remote_path)
        except:
            pass

        if os.path.isdir(local_path):
            for f in os.listdir(local_path):
                local_file = os.path.join(local_path, f)
                remote_file = remote_path + "/" + f
                if os.path.isdir(local_file):
                    self._put_recursive(local_file, remote_file)
                else:
                    self.sftp.put(local_file, remote_file)
                    print(f"Uploaded {local_file}")
        else:
            self.sftp.put(local_path, remote_path)
            print(f"Uploaded {local_path}")

    def _mkdir_remote(self, remote_path):
        try:
            self.sftp.mkdir(remote_path)
        except IOError:
            pass  # directory exists

    # -------------------
    # Custom commands
    # -------------------
    def do_backup_txt(self, arg):
        for f in os.listdir("."):
            if f.endswith(".txt"):
                self.sftp.put(f, f)
                print(f"Uploaded {f}")
                
    def do_lcd(self, path):
        """Change local working directory"""
        if not path:
            print(os.getcwd())
            return

        try:
            os.chdir(path)
            print(f"Local directory now: {os.getcwd()}")
        except FileNotFoundError:
            print(f"No such directory: {path}")
        except NotADirectoryError:
            print(f"{path} is not a directory")

    # Exit
    def do_exit(self, arg):
        return True

    do_EOF = do_exit

    # Piped commands support
    def run_from_pipe(self):
        if not sys.stdin.isatty():
            for line in sys.stdin:
                line = line.strip()
                if line:
                    self.onecmd(line)
        else:
            self.cmdloop()


# -------------------
# Main entry
# -------------------
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--hostname",type=str,default="chip-login1.rs.umbc.edu")
    parser.add_argument("--username",type=str, default="jbaker15")
    args=parser.parse_args()
    hostname = args.hostname
    username = args.username
    keyfile = input("SSH key file (leave blank for default ~/.ssh/id_rsa): ").strip() or None

    shell = SFTPShell(hostname, username, key_filename=keyfile)
    shell.run_from_pipe()