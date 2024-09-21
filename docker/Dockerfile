# adapted from https://docs.docker.com/engine/examples/running_ssh_service/
FROM python

# Install OpenSSH server and essential tools (gcc, cmake, git)
RUN apt-get update && apt-get install -y \
    openssh-server \
    gcc \
    cmake \
    git

# Create SSH directory and set root password
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd

# Enable root login via SSH
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# SSH login fix. Otherwise, the user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# Set environment variable
ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# Expose SSH port
EXPOSE 22

# Start SSH service
CMD ["/usr/sbin/sshd", "-D"]
